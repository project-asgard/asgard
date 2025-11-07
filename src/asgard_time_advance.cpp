#include "asgard_discretization.hpp"

#include "asgard_small_mats.hpp"

namespace asgard::time_advance
{

template<typename P>
void steady_state<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &endstep) const
{
  tools::time_event performance_("solve steady state");

  P const time = disc.stop_time();

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  if (solver.grid_gen != disc.grid_generation())
    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), 0);

  if (solver.opt == solver_method::direct) {

    endstep.resize(current.size());
    disc.set_ode_rhs_sources(time, 1, endstep);

    if (disc.is_leader())
      solver.direct_solve(endstep);

  } else { // iterative solver
    // form the right-hand-side inside work
    endstep = current; // initial guess

    int64_t const n = static_cast<int64_t>(current.size());

    work.resize(n);
    disc.set_ode_rhs_sources(time, 1, work); // right-hand-side

    if (not disc.is_leader()) {
      // enter worker mode for iterative solver
      disc.mpi_iteration_apply(work);
      return;
    }

    switch (solver.precon) {
    case precon_method::none:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      ignore(n);
      t1 = work;
      t2 = work;
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.terms_apply_gpu(alpha, x, beta, y);
        }, t1, t2);
      t2.copy_to_host(endstep);
      #else
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply(alpha, x, beta, y);
        }, work, endstep);
      #endif
    break;
    case precon_method::jacobi:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      t1 = work;
      t2 = work;
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(solver.jacobi_gpu, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.terms_apply_gpu(alpha, x, beta, y);
        }, t1, t2);
      t2.copy_to_host(endstep);
      #else
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, solver.jacobi, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply(alpha, x, beta, y);
        }, work, endstep);
      #endif
    break;
    default:
      throw std::runtime_error("steady state solver cannot use the adi preconditioner");
    }

    disc.mpi_iteration_stop();
  }
}

template<typename P>
void rungekutta<P>::leader_sum(discretization_manager<P> const &disc,
                               std::vector<P> const &x, P a1, std::vector<P> const &x1,
                               std::vector<P> &y)
{
  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = x[i] + a1 * x1[i];
  }
}
template<typename P>
void rungekutta<P>::leader_sum(discretization_manager<P> const &disc,
                               std::vector<P> const &x, P a1, std::vector<P> const &x1,
                               P a2, std::vector<P> const &x2,
                               std::vector<P> &y)
{
  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = x[i] + a1 * x1[i] + a2 * x2[i];
  }
}
template<typename P>
void rungekutta<P>::leader_sum(discretization_manager<P> const &disc,
                               std::vector<P> const &x, P a1, std::vector<P> const &x1,
                               P a2, std::vector<P> const &x2, P a3, std::vector<P> const &x3,
                               std::vector<P> &y)
{
  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = x[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i];
  }
}
template<typename P>
void rungekutta<P>::leader_sum(discretization_manager<P> const &disc,
                               std::vector<P> const &x, P a1, std::vector<P> const &x1,
                               P a2, std::vector<P> const &x2, P a3, std::vector<P> const &x3,
                               P a4, std::vector<P> const &x4, std::vector<P> &y)
{
  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < x.size(); i++)
      y[i] = x[i] + a1 * x1[i] + a2 * x2[i] + a3 * x3[i] + a4 * x4[i];
  }
}

template<typename P>
void rungekutta<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &next) const
{
  std::string const name = [&]() -> std::string {
      switch (rktype) {
        case time_method::forward_euler:
          return "forw-euler";
        case time_method::rk2:
          return "runge kutta 2";
        case time_method::rk3:
          return "runge kutta 3";
        default: // case method::rk4:
          return "runge kutta 4";
      };
    }();

  tools::time_event performance_(name);

  P const time = disc.time();
  P const dt   = disc.dt();

  switch (rktype) {
    case time_method::forward_euler:
      disc.ode_euler(time, current, dt, next);
      break;
    case time_method::rk2:
      disc.ode_euler(time, current, 0.5 * dt, s1);
      disc.ode_rhs(time + 0.5 * dt, s1, k1);

      next.resize(current.size());
      leader_sum(disc, current, dt, k1, next);
      break;
    case time_method::rk3:
      s1.resize(current.size());
      next.resize(current.size());

      disc.ode_rhs(time, current, k1);
      leader_sum(disc, current, 0.5 * dt, k1, s1);

      disc.ode_rhs(time + 0.5 * dt, s1, k2);
      leader_sum(disc, current, -dt, k1, 2 * dt, k2, s1);

      disc.ode_rhs(time + dt, s1, k3);
      {
        P const dt6 = dt / P{6};
        leader_sum(disc, current, dt6, k1, 4 * dt6, k2, dt6, k3, next);
      }
      break;
    case time_method::rk4:
      s1.resize(current.size());
      next.resize(current.size());

      disc.ode_rhs(time, current, k1);
      leader_sum(disc, current, 0.5 * dt, k1, s1);

      disc.ode_rhs(time + 0.5 * dt, s1, k2);
      leader_sum(disc, current, 0.5 * dt, k2, s1);

      disc.ode_rhs(time + 0.5 * dt, s1, k3);
      leader_sum(disc, current, dt, k3, s1);

      disc.ode_rhs(time + dt, s1, k4);
      {
        P const dt6 = dt / P{6};
        leader_sum(disc, current, dt6, k1, 2 * dt6, k2, 2 * dt6, k3, dt6, k4, next);
      }
      break;
    default: // unreachable
      expect(false); // should never get here
      break;
  }
}

template<typename P>
void crank_nicolson<P>::set_rhs(discretization_manager<P> const &disc, P substep, P time, P dt,
                                std::vector<P> const &current, std::vector<P> &rhs) const
{
  if (substep == 1)
    disc.ode_euler(time + substep * dt, current, terms_scale{0}, sources_scale{dt}, rhs);
  else
    disc.ode_euler(time + substep * dt, current,
                   terms_scale{dt * (1 - substep)}, sources_scale{dt}, rhs);
}

template<typename P>
void crank_nicolson<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &next) const
{
  tools::time_event performance_(
      (method == time_method::cn) ? "crank-nicolson" : "back-euler");

  P const time = disc.time();
  P const dt   = disc.dt();

  P const substep = (method == time_method::cn) ? 0.5 : 1;

  if (disc.has_moments())
    disc.compute_moments(current);

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  if (solver.grid_gen != disc.grid_generation())
    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt);

  if (solver.opt == solver_method::direct) {

    next.resize(current.size());
    set_rhs(disc, substep, time, dt, current, next);

    if (disc.is_leader())
      solver.direct_solve(next);

  } else { // iterative solver
    // form the right-hand-side inside work
    work = current;

    set_rhs(disc, substep, time, dt, current, work);

    next = current; // use the current step as the initial guess

    int64_t const n = static_cast<int64_t>(work.size());

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply(work);
      return;
    }

    switch (solver.precon) {
    case precon_method::none:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      ignore(n);
      t1 = work;
      t2 = current;
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(t1.size(), alpha, x, beta, y);
          disc.terms_apply_gpu(substep * alpha * dt, x, 1, y);
        }, t1, t2);
      t2.copy_to_host(next);
      #else
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(substep * alpha * dt, x, 1, y);
        }, work, next);
      #endif
    break;
    case precon_method::jacobi:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      t1 = work;
      t2 = current;
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(solver.jacobi_gpu, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(t1.size(), alpha, x, beta, y);
          disc.terms_apply_gpu(substep * alpha * dt, x, 1, y);
        }, t1, t2);
      t2.copy_to_host(next);
      #else
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, solver.jacobi, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(substep * alpha * dt, x, 1, y);
        }, work, next);
      #endif
    break;
    default:
    break;
    }

    disc.mpi_iteration_stop();
  }
}

template<typename P>
void imex_stepper<P>::implicit_solve(
    discretization_manager<P> const &disc, P time,
    std::vector<P> &current, std::vector<P> &R) const
{
  if (disc.has_moments())
    disc.compute_moments(group_id{imex_implicit}, current);

  P const dt = disc.dt();

  solver.update_grid(imex_implicit.gid, disc.get_grid(), disc.get_conn(),
                     disc.get_terms(), dt);

  if (solver.opt != solver_method::direct)
    R = current;

  disc.add_ode_rhs_sources_group(group_id{imex_implicit}, time, dt, current);

  if (solver.opt == solver_method::direct) {
    R = current; // copy
    solver.direct_solve(R);
  } else { // iterative solver
    int64_t const n = static_cast<int64_t>(R.size());

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply(group_id{imex_implicit}, current);
      return;
    }

    switch (solver.precon) {
    case precon_method::none:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      ignore(n);
      t1 = current;
      t2 = R;
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(t1.size(), alpha, x, beta, y);
          disc.terms_apply_gpu(group_id{imex_implicit.gid}, alpha * dt, x, 1, y);
        }, t1, t2);
      t2.copy_to_host(R);
      #else
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(group_id{imex_implicit.gid}, alpha * dt, x, 1, y);
        }, current, R);
      #endif
    break;
    case precon_method::jacobi:
      #if defined(ASGARD_USE_GPU) && !defined(ASGARD_USE_MPI)
      t1 = current;
      t2 = R;
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(solver.jacobi_gpu, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(t1.size(), alpha, x, beta, y);
          disc.terms_apply_gpu(group_id{imex_implicit.gid}, alpha * dt, x, 1, y);
        }, t1, t2);
      t2.copy_to_host(R);
      #else
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, solver.jacobi, y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(group_id{imex_implicit.gid}, alpha * dt, x, 1, y);
        }, current, R);
      #endif
    break;
    default:
      throw std::runtime_error("adi preconditioner not available for IMEX steppers");
    break;
    }
  }

  disc.mpi_iteration_stop();
}

template<typename P>
void imex_stepper<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &next) const
{
  tools::time_event performance_("stepper-imex");

  P const time = disc.time();
  P const dt   = disc.dt();

  disc.ode_euler(group_id{imex_explicit}, time, current, dt, f);

  implicit_solve(disc, time + dt, f, next);

  if (method == time_method::imex1)
    return;

  disc.ode_rhs(group_id{imex_explicit}, time + dt, next, f);

  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < f.size(); i++)
      f[i] = 0.5 * current[i] + 0.5 * (next[i] + dt * f[i]);
  }

  implicit_solve(disc, time + dt, f, next);
}

}

namespace asgard
{

template<typename P>
time_advance_manager<P>::time_advance_manager(time_data const &tdata, prog_opts const &options)
  : data(tdata)
{
  // prepare the time-stepper
  switch (data.step_method())
  {
    case time_method::steady:
      method = time_advance::steady_state<P>(options);
      break;
    case time_method::forward_euler:
    case time_method::rk2:
    case time_method::rk3:
    case time_method::rk4:
      method = time_advance::rungekutta<P>(data.step_method());
      break;
    case time_method::cn:
    case time_method::back_euler:
      method = time_advance::crank_nicolson<P>(options);
      break;
    case time_method::imex1:
    case time_method::imex2:
      throw std::runtime_error("invalid constructor for an imex method");
    default:
      // should be unreachable
      throw std::runtime_error("invalid time-advance option");
  }
}

template<typename P>
time_advance_manager<P>::time_advance_manager(
    time_data const &tdata, prog_opts const &options,
    imex_implicit_group im, imex_explicit_group ex)
    : data(tdata)
{
  expect(is_imex(data.step_method()));
  rassert(im.gid >= -1 and ex.gid >= -1,
          "the IMEX implicit and explicit groups have not been set in the pde_scheme");

  method = time_advance::imex_stepper<P>(options, im, ex);
}

template<typename P>
void time_advance_manager<P>::next_step(discretization_manager<P> const &dist,
                                        std::vector<P> const &current,
                                        std::vector<P> &next) const
{
  switch (method.index()) {
    case 0: // steady state
      std::get<0>(method).next_step(dist, current, next);
      break;
    case 1: // explicit rk
      std::get<1>(method).next_step(dist, current, next);
      break;
    case 2: // implicit stepper
      std::get<2>(method).next_step(dist, current, next);
      break;
    case 3: // imex stepper
      std::get<3>(method).next_step(dist, current, next);
      break;
    default:
      // this should be unreachable
      throw std::runtime_error("internal error, invalid time-advance method");
  };
}

template<typename P> // implemented in time-advance
void advance_in_time(discretization_manager<P> &manager, int64_t num_steps)
{
  // periodically reports time, first initialization is not important
  static tools::simple_timer::time_point wctime = tools::simple_timer::current_time();
  wctime = tools::simple_timer::current_time(); // initialization for this call to advance_in_time()

  time_advance_manager<P> const &stepper = manager.stepper;

  time_data &params = manager.stepper.data;

  // is num_steps is negative, run to the end of num_remain()
  // otherwise, run num_steps but no more than num_remain()
  if (num_steps > 0)
    num_steps = std::min(params.num_remain(), num_steps);
  else
    num_steps = std::max(params.num_remain(), num_steps);

  if (stepper.is_steady_state())
    num_steps = 1;

  if (num_steps < 1)
    return;

  P const atol = manager.options().adapt_threshold.value_or(0);
  P const rtol = manager.options().adapt_relative.value_or(0);

  sparse_grid &grid = manager.grid;

  sparse_grid::strategy grid_strategy = sparse_grid::strategy::refine;

  std::vector<P> next;
  while (--num_steps >= 0)
  {
    stepper.next_step(manager, manager.state, next);

    if (atol > 0 or rtol > 0) {
      int const gen = grid.generation();
      if (manager.is_leader()) {
        grid.refine(atol, rtol, manager.hier.block_size(),
                    manager.conn[connect_1d::hierarchy::volume], grid_strategy, next);
      }
      manager.grid_sync(); // no-op, unless MPI or GPUs are enabled
      if (grid.generation() != gen) {
        if (manager.is_leader())
          grid.remap(manager.hier.block_size(), next);
        manager.terms.prapare_kron_workspace(grid);
        if (manager.poisson)
          manager.poisson.update_level(grid.current_level(0));
        if (stepper.is_steady_state()) {
          num_steps = 1;
          grid_strategy = sparse_grid::strategy::adapt;
        }
      }
    }

    #ifdef ASGARD_USE_MPI
    if (manager.is_leader())
      std::swap(manager.state, next);
    else
      manager.state.resize(grid.num_indexes() * manager.get_hier().block_size());
    #else
    std::swap(manager.state, next);
    #endif

    params.take_step();

    if (not manager.stop_verbosity()) {
      // if verbosity is not turned off, report every 2 or 10 seconds
      double duration = tools::simple_timer::duration_since(wctime);
      if ((manager.high_verbosity() and duration > 2000) or (duration > 10000)) {
        manager.progress_report();
        wctime = tools::simple_timer::current_time();
      }
    }

    if (stepper.is_steady_state())
      params.set_final_time();
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct time_advance::steady_state<double>;
template struct time_advance::rungekutta<double>;
template struct time_advance::crank_nicolson<double>;
template struct time_advance_manager<double>;

template void advance_in_time(discretization_manager<double> &, int64_t);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct time_advance::steady_state<float>;
template struct time_advance::rungekutta<float>;
template struct time_advance::crank_nicolson<float>;
template struct time_advance_manager<float>;

template void advance_in_time(discretization_manager<float> &, int64_t);
#endif
} // namespace asgard
