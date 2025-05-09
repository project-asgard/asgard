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

    solver.direct_solve(endstep);

  } else { // iterative solver
    // form the right-hand-side inside work
    endstep = current; // initial guess

    int64_t const n = static_cast<int64_t>(current.size());

    work.resize(n);
    disc.set_ode_rhs_sources(time, 1, work); // right-hand-side

    switch (solver.precon) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.terms_apply_all(alpha, x, beta, y);
        }, work, endstep);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] *= solver.jacobi[i];
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.terms_apply_all(alpha, x, beta, y);
        }, work, endstep);
    break;
    default:
      throw std::runtime_error("steady state solver cannot use the adi preconditioner");
    }
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

  if (disc.has_moments() and not disc.has_poisson())
    disc.compute_moments(current);

  switch (rktype) {
    case time_method::forward_euler:
      k1.resize(current.size());
      disc.ode_rhs(time, current, k1);

      next.resize(current.size());

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        next[i] = current[i] + dt * k1[i];
      break;
    case time_method::rk2:
      k1.resize(current.size());
      k2.resize(current.size());
      s1.resize(current.size());

      disc.ode_rhs(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs(time + 0.5 * dt, s1, k2);

      next.resize(current.size());

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        next[i] = current[i] + dt * k2[i];
      break;
    case time_method::rk3:
      k1.resize(current.size());
      k2.resize(current.size());
      k3.resize(current.size());
      s1.resize(current.size());

      disc.ode_rhs(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs(time + 0.5 * dt, s1, k2);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] - dt * k1[i] + 2 * dt * k2[i];

      disc.ode_rhs(time + dt, s1, k3);

      next.resize(current.size());

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        next[i] = current[i] + dt * (k1[i] + 4 * k2[i] + k3[i]) / P{6};
      break;
    case time_method::rk4:
      k1.resize(current.size());
      k2.resize(current.size());
      k3.resize(current.size());
      k4.resize(current.size());
      s1.resize(current.size());

      disc.ode_rhs(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs(time + 0.5 * dt, s1, k2);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k2[i];

      disc.ode_rhs(time + 0.5 * dt, s1, k3);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + dt * k3[i];

      disc.ode_rhs(time + dt, s1, k4);

      next.resize(current.size());

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        next[i] = current[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / P{6};
      break;
    default: // unreachable
      expect(false); // should never get here
      break;
  }
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

  if (disc.has_moments() and not disc.has_poisson()) {
    disc.compute_moments(current);
  }

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  if (solver.grid_gen != disc.grid_generation())
    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt);

  if (solver.opt == solver_method::direct) {
    next = current; // copy

    if (substep < 1)
      disc.terms_apply_all(-substep * dt, current, 1, next);
    disc.add_ode_rhs_sources(time + substep * dt, dt, next);

    solver.direct_solve(next);
  } else { // iterative solver
    // form the right-hand-side inside work
    work = current;
    if (substep < 1)
      disc.terms_apply_all(-substep * dt, current, 1, work);
    disc.add_ode_rhs_sources(time + substep * dt, dt, work);

    next = current; // use the current step as the initial guess

    int64_t const n = static_cast<int64_t>(work.size());

    switch (solver.precon) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] = alpha * x[i] + beta * y[i];
          disc.terms_apply_all(substep * alpha * dt, x, 1, y);
        }, work, next);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] *= solver.jacobi[i];
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] = alpha * x[i] + beta * y[i];
          disc.terms_apply_all(substep * alpha * dt, x, 1, y);
        }, work, next);
    break;
    default: {
      static std::vector<P> adi_work;
      adi_work.resize(work.size());
      // assuming ADI
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          disc.terms_apply_adi(y, adi_work.data());
          std::copy(adi_work.begin(), adi_work.end(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] = alpha * x[i] + beta * y[i];
          disc.terms_apply_all(substep * alpha * dt, x, 1, y);
        }, work, next);
    }
    break;
    }
  }
}

template<typename P>
void imex_stepper<P>::explicit_ode_rhs(
    discretization_manager<P> const &disc, P time, std::vector<P> const &current,
    std::vector<P> &R) const
{
  disc.compute_poisson(imex_explicit.gid, current);
  disc.compute_moments(imex_explicit.gid, current);

  if (R.size() != current.size())
    R.resize(current.size());

  disc.terms_apply(imex_explicit.gid, -1, current, 0, R);
  disc.add_ode_rhs_sources_group(imex_explicit.gid, time, R);
}
template<typename P>
void imex_stepper<P>::implicit_solve(
    discretization_manager<P> const &disc, P time,
    std::vector<P> &current, std::vector<P> &R) const
{
  disc.compute_moments(imex_implicit.gid, current);

  P const dt = disc.dt();

  solver.update_grid(imex_implicit.gid, disc.get_grid(), disc.get_conn(),
                     disc.get_terms(), dt);

  disc.add_ode_rhs_sources_group(imex_implicit.gid, time, dt, current);

  if (solver.opt == solver_method::direct) {
    R = current; // copy
    solver.direct_solve(R);
  } else { // iterative solver
    // form the right-hand-side inside work
    R = current;

    int64_t const n = static_cast<int64_t>(R.size());

    switch (solver.precon) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] = alpha * x[i] + beta * y[i];
          disc.terms_apply(imex_implicit.gid, alpha * dt, x, 1, y);
        }, current, R);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] *= solver.jacobi[i];
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          ASGARD_OMP_PARFOR_SIMD
          for (int64_t i = 0; i < n; i++)
            y[i] = alpha * x[i] + beta * y[i];
          disc.terms_apply(imex_implicit.gid, alpha * dt, x, 1, y);
        }, current, R);
    break;
    default:
      throw std::runtime_error("adi preconditioner not available for IMEX steppers");
    break;
    }
  }
}

template<typename P>
void imex_stepper<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &next) const
{
  tools::time_event performance_("stepper-imex");

  P const time = disc.time();
  P const dt   = disc.dt();

  explicit_ode_rhs(disc, time, current, fs);

  f.resize(fs.size());

  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < current.size(); i++)
    f[i] = current[i] + dt * fs[i];

  implicit_solve(disc, time + dt, f, fs);

  explicit_ode_rhs(disc, time + dt, fs, f);

  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < f.size(); i++)
    f[i] = 0.5 * current[i] + 0.5 * (fs[i] + dt * f[i]);

  implicit_solve(disc, time + dt, f, next);
}

}

namespace asgard
{

template<typename P>
time_advance_manager<P>::time_advance_manager(time_data const &tdata, prog_opts const &options)
  : data(tdata)
{
  expect(static_cast<int>(data.step_method()) <= 6); // the new modes that have been implemented

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
    default:
      throw std::runtime_error("unimplemented time-advance option");
  }
}

template<typename P>
time_advance_manager<P>::time_advance_manager(
    time_data const &tdata, prog_opts const &options,
    imex_implicit_group im, imex_explicit_group ex)
    : data(tdata)
{
  expect(is_imex(data.step_method()));

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
      throw std::runtime_error("unimplemented time-advance option");
  };
}

template<typename P> // implemented in time-advance
void advance_in_time(discretization_manager<P> &manager, int64_t num_steps)
{
  // periodically reports time
  static tools::simple_timer::time_point wctime = tools::simple_timer::current_time();

  time_advance_manager<P> const &stepper = manager.stepper;

  time_data &params = manager.stepper.data;

  // is num_steps is negative, run to the end of num_remain()
  // otherwise, run num_steps but no more than num_remain()
  if (num_steps > 0)
    num_steps = std::min(params.num_remain(), num_steps);
  else
    num_steps = std::max(params.num_remain(), num_steps);

  if (num_steps < 1)
    return;

  if (stepper.is_steady_state())
    num_steps = 1;

  P const atol = manager.options().adapt_threshold.value_or(0);
  P const rtol = manager.options().adapt_ralative.value_or(0);

  sparse_grid &grid = manager.grid;

  sparse_grid::strategy grid_strategy = sparse_grid::strategy::refine;

  std::vector<P> next;
  while (--num_steps >= 0)
  {
    stepper.next_step(manager, manager.state, next);

    if (atol > 0 or rtol > 0) {
      int const gen = grid.generation();
      grid.refine(atol, rtol, manager.hier.block_size(),
                  manager.conn[connect_1d::hierarchy::volume], grid_strategy, next);
      if (grid.generation() != gen) {
        grid.remap(manager.hier.block_size(), next);
        manager.terms.prapare_workspace(grid);
        if (stepper.is_steady_state()) {
          num_steps = 1;
          grid_strategy = sparse_grid::strategy::adapt;
        }
      }
    }

    std::swap(manager.state, next);

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
