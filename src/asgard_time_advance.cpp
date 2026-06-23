#include "asgard_discretization.hpp"

#include "asgard_small_mats.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

// see the comment in asgard_refinement.cpp
#define ASGARD_INFINITE_TRESHOLD 1.E+100

namespace asgard::time_advance
{

inline std::string toMB(size_t bytes) {
  std::string s = std::to_string(bytes / (1024 * 1024)) + "MB\n";
  s.insert(0, 11 - s.size(), ' ');
  return s;
};

template<typename P>
void steady_state<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &endstep) const
{
  #ifdef ASGARD_USE_GPU
  gcurrent = current;
  gnext.resize(gcurrent.size());
  next_step(disc, gcurrent, gnext);
  gnext.copy_to_host(endstep);
  return;
  #endif

  tools::time_event performance_("solve steady state");

  P const time = disc.stop_time();

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), 0, precon);

  if (solver.uses_inplace_solve()) {

    endstep.resize(current.size());
    disc.set_ode_rhs_sources(time, 1, endstep);

    if (disc.is_leader())
      solver.solve_inplace(endstep);

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

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply(alpha, x, beta, y);
        }, work, endstep);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, precon.jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply(alpha, x, beta, y);
        }, work, endstep);
    break;
    default:
      throw std::runtime_error("steady state solver cannot use the adi preconditioner");
    }

    disc.mpi_iteration_stop();
  }
}

#ifdef ASGARD_USE_GPU
template<typename P>
void steady_state<P>::next_step(discretization_manager<P> const &disc,
                                gpu::vector<P> const &current,
                                gpu::vector<P> &endstep) const
{
  tools::time_event performance_("solve steady state-gpu");

  P const time = disc.stop_time();

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), 0, precon);

  if (solver.uses_inplace_solve()) {

    endstep.resize(current.size());

    disc.set_ode_rhs_sources_group_gpu(group_id::all(), time, endstep.data());

    if (disc.is_leader())
      solver.solve_inplace(endstep.data());

  } else { // iterative solver
    // form the right-hand-side inside work
    int64_t const num_entries = disc.num_dof();

    endstep = current;

    gwork.resize(num_entries);
    disc.set_ode_rhs_sources_group_gpu(group_id::all(), time, gwork.data()); // right-hand-side

    if (not disc.is_leader()) {
      // enter worker mode for iterative solver
      disc.mpi_iteration_apply_gpu(gwork.data());
      return;
    }

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply_gpu(alpha, x, beta, y);
        }, gwork, endstep);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(precon.gpu_jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          disc.mpi_leader_apply_gpu(alpha, x, beta, y);
        }, gwork, endstep);
    break;
    default:
      throw std::runtime_error("steady state solver cannot use the adi preconditioner");
    }

    disc.mpi_iteration_stop_gpu();
  }
}
#endif

template<typename P>
void steady_state<P>::print_bytes(std::ostream &os) const {
  os << "time-advance\n";
  os << "  workspace " << toMB(work.size() * sizeof(P)) << '\n';
  os << "  solver    " << toMB(solver.used_bytes()) << '\n';
  os << "  precon    " << toMB(precon.used_bytes()) << '\n';
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
  #ifdef ASGARD_USE_GPU
  gcurrent = current;
  next_step(disc, gcurrent, gnext);
  gnext.copy_to_host(next);
  return;
  #endif

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
      assert(false); // should never get here
      break;
  }
}

#ifdef ASGARD_USE_GPU
template<typename P>
void rungekutta<P>::next_step(
    discretization_manager<P> const &disc, gpu::vector<P> const &current,
    gpu::vector<P> &next) const
{
  std::string const name = [&]() -> std::string {
      switch (rktype) {
        case time_method::forward_euler:
          return "forw-euler gpu";
        case time_method::rk2:
          return "runge kutta 2 gpu";
        case time_method::rk3:
          return "runge kutta 3 gpu";
        default: // case method::rk4:
          return "runge kutta 4 gpu";
      };
    }();

  tools::time_event performance_(name);

  next.resize(disc.num_dof());
  assert(next.size() == current.size());

  P const time = disc.time();
  P const dt   = disc.dt();

  switch (rktype) {
    case time_method::forward_euler:
      disc.ode_euler_gpu(time, current.data(), dt, next.data());
      break;
    case time_method::rk2:
      gs1.resize(next.size());
      gk1.resize(gs1.size());
      disc.ode_euler_gpu(time, current.data(), 0.5 * dt, gs1.data());
      disc.ode_rhs_gpu(time + 0.5 * dt, gs1.data(), gk1.data());

      if (disc.is_leader())
        gpu::sum2(current, dt, gk1, next);
      break;
    case time_method::rk3:
      gs1.resize(next.size());
      gk1.resize(gs1.size());
      gk2.resize(gs1.size());
      gk3.resize(gs1.size());

      disc.ode_rhs_gpu(time, current.data(), gk1.data());
      if (disc.is_leader())
        gpu::sum2(current, 0.5 * dt, gk1, gs1);

      disc.ode_rhs_gpu(time + 0.5 * dt, gs1.data(), gk2.data());
      if (disc.is_leader())
        gpu::sum3(current, -dt, gk1, 2 * dt, gk2, gs1);

      disc.ode_rhs_gpu(time + dt, gs1.data(), gk3.data());
      {
        P const dt6 = dt / P{6};
        if (disc.is_leader())
          gpu::sum4(current, dt6, gk1, 4 * dt6, gk2, dt6, gk3, next);
      }
      break;
    case time_method::rk4:
      gs1.resize(next.size());
      gk1.resize(gs1.size());
      gk2.resize(gs1.size());
      gk3.resize(gs1.size());
      gk4.resize(gs1.size());

      disc.ode_rhs_gpu(time, current.data(), gk1.data());
      if (disc.is_leader())
        gpu::sum2(current, 0.5 * dt, gk1, gs1);

      disc.ode_rhs_gpu(time + 0.5 * dt, gs1.data(), gk2.data());
      if (disc.is_leader())
        gpu::sum2(current, 0.5 * dt, gk2, gs1);

      disc.ode_rhs_gpu(time + 0.5 * dt, gs1.data(), gk3.data());
      if (disc.is_leader())
        gpu::sum2(current, dt, gk3, gs1);

      disc.ode_rhs_gpu(time + dt, gs1.data(), gk4.data());
      {
        P const dt6 = dt / P{6};
        if (disc.is_leader())
          gpu::sum5(current, dt6, gk1, 2 * dt6, gk2, 2 * dt6, gk3, dt6, gk4, next);
      }
      break;
    default: // unreachable
      assert(false); // should never get here
      break;
  }
}
#endif

template<typename P>
void rungekutta<P>::print_bytes(std::ostream &os) const {
  size_t const t = k1.size() + k2.size() + k3.size() + k4.size() + s1.size();
  os << "time-advance\n";
  os << "  workspace " << toMB(t * sizeof(P)) << '\n';
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
  #ifdef ASGARD_USE_GPU
  gcurrent = current;
  gnext.resize(gcurrent.size());
  next_step(disc, gcurrent, gnext);
  gnext.copy_to_host(next);
  return;
  #endif

  tools::time_event performance_(
      (method == time_method::cn) ? "crank-nicolson" : "back-euler");

  P const time = disc.time();
  P const dt   = disc.dt();

  P const substep = (method == time_method::cn) ? 0.5 : 1;

  if (solver.uses_inplace_solve()) {

    next.resize(current.size());
    set_rhs(disc, substep, time, dt, current, next);

    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt, precon);

    if (disc.is_leader())
      solver.solve_inplace(next);

  } else { // iterative solver
    // form the right-hand-side inside work
    work.resize(current.size());

    set_rhs(disc, substep, time, dt, current, work);

    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt, precon);

    next = current; // use the current step as the initial guess

    int64_t const n = static_cast<int64_t>(work.size());

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply(work);
      return;
    }

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(substep * alpha * dt, x, 1, y);
        }, work, next);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, precon.jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(substep * alpha * dt, x, 1, y);
        }, work, next);
    break;
    default:
    break;
    }

    disc.mpi_iteration_stop();
  }
}

#ifdef ASGARD_USE_GPU
template<typename P>
void crank_nicolson<P>::set_rhs_gpu(discretization_manager<P> const &disc, P substep, P time, P dt,
                                    gpu::vector<P> const &current, gpu::vector<P> &rhs) const
{
  assert(current.size() == disc.num_dof());
  rhs.resize(current.size());
  if (substep == 1)
    disc.ode_euler_gpu(time + substep * dt, current.data(),
                       terms_scale{0}, sources_scale{dt}, rhs.data());
  else
    disc.ode_euler_gpu(time + substep * dt, current.data(),
                       terms_scale{dt * (1 - substep)}, sources_scale{dt}, rhs.data());
}

template<typename P>
void crank_nicolson<P>::next_step(discretization_manager<P> const &disc,
                                  gpu::vector<P> const &current,
                                  gpu::vector<P> &next) const
{
  tools::time_event performance_(
      (method == time_method::cn) ? "crank-nicolson-gpu" : "back-euler-gpu");

  P const time = disc.time();
  P const dt   = disc.dt();

  P const substep = (method == time_method::cn) ? 0.5 : 1;

  if (solver.uses_inplace_solve()) {

    set_rhs_gpu(disc, substep, time, dt, current, next);

    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt, precon);

    if (disc.is_leader())
      solver.solve_inplace(next.data());

  } else { // iterative solver
    // form the right-hand-side inside work
    int64_t const num_entries = disc.num_dof();
    gwork.resize(num_entries);

    set_rhs_gpu(disc, substep, time, dt, current, gwork);

    solver.update_grid(disc.get_grid(), disc.get_conn(), disc.get_terms(), substep * dt, precon);

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply_gpu(gwork.data());
      return;
    }

    // use the current step as the initial guess
    next = current;

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(num_entries, alpha, x, beta, y);
          disc.mpi_leader_apply_gpu(substep * alpha * dt, x, 1, y);
        }, gwork, next);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(precon.gpu_jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(num_entries, alpha, x, beta, y);
          disc.mpi_leader_apply_gpu(substep * alpha * dt, x, 1, y);
        }, gwork, next);
    break;
    default:
    break;
    }

    disc.mpi_iteration_stop_gpu();
  }
}
#endif

template<typename P>
void crank_nicolson<P>::print_bytes(std::ostream &os) const {
  os << "time-advance\n";
  os << "  workspace " << toMB(work.size() * sizeof(P)) << '\n';
  os << "  solver    " << toMB(solver.used_bytes()) << '\n';
  os << "  precon    " << toMB(precon.used_bytes()) << '\n';
}

template<typename P>
void imex_stepper<P>::implicit_solve(
    discretization_manager<P> const &disc, size_t stage,
    P time, P dt, preconditioner_data<P> &precon,
    std::vector<P> &current, std::vector<P> &R) const
{
  if (disc.has_moments())
    disc.compute_moments(group_id{imex_implicit}, current);

  solver.update_grid(group_id{imex_implicit}, stage, disc.get_grid(), disc.get_conn(),
                     disc.get_terms(), dt, precon);

  bool const uses_inplace = solver.uses_inplace_solve();
  if (not uses_inplace)
    R = current; // save current before it is updated below

  disc.add_ode_rhs_sources_group(group_id{imex_implicit}, time, dt, current);

  if (uses_inplace) {
    R = current;
    solver.solve_inplace(group_id{imex_implicit}, stage, R);
  } else { // iterative solver
    int64_t const n = static_cast<int64_t>(R.size());

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply(group_id{imex_implicit}, current);
      return;
    }

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(group_id{imex_implicit}, alpha * dt, x, 1, y);
        }, current, R);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          fm::jacobi_apply(n, precon.jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          fm::axpby(n, alpha, x, beta, y);
          disc.mpi_leader_apply(group_id{imex_implicit}, alpha * dt, x, 1, y);
        }, current, R);
    break;
    default:
      throw std::runtime_error("adi preconditioner not available for IMEX steppers");
    break;
    }

    disc.mpi_iteration_stop();
  }
}

template<typename P>
void imex_stepper<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &next) const
{
  #ifdef ASGARD_USE_GPU
  gcurrent = current;
  gnext.resize(gcurrent.size());
  next_step(disc, gcurrent, gnext);
  gnext.copy_to_host(next);
  return;
  #endif

  tools::time_event performance_("stepper-imex");

  P const time = disc.time();
  P const dt   = disc.dt();

  disc.ode_euler(group_id{imex_explicit}, time, current, dt, f);

  constexpr size_t stage0 = 0;
  implicit_solve(disc, stage0, time + dt, dt, precon1, f, next);

  if (method == time_method::imex1)
    return;

  disc.ode_rhs(group_id{imex_explicit}, time + dt, next, f);

  if (disc.is_leader()) {
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < f.size(); i++)
      f[i] = 0.5 * current[i] + 0.5 * (next[i] + dt * f[i]);
  }

  constexpr size_t stage1 = 1;
  implicit_solve(disc, stage1, time + dt, P{0.5} * dt, precon2, f, next);
}

#ifdef ASGARD_USE_GPU
template<typename P>
void imex_stepper<P>::implicit_solve(
    discretization_manager<P> const &disc, size_t stage,
    P time, P dt, preconditioner_data<P> &precon, gpu::vector<P> &current, gpu::vector<P> &R) const
{
  int64_t const num_entries = disc.num_dof();
  if (disc.has_moments())
    disc.compute_moments_gpu(group_id{imex_implicit}, current.data());

  solver.update_grid(group_id{imex_implicit}, stage, disc.get_grid(), disc.get_conn(),
                     disc.get_terms(), dt, precon);

  bool const uses_inplace = solver.uses_inplace_solve();
  if (not uses_inplace)
    R = current; // save current as the initial guess

  disc.add_ode_rhs_sources_group_gpu(group_id{imex_implicit}, time, dt, current.data());

  if (uses_inplace) {
    R = current;
    solver.solve_inplace(group_id{imex_implicit}, stage, R.data());
  } else { // iterative solver

    if (not disc.is_leader()) {
      disc.mpi_iteration_apply_gpu(group_id{imex_implicit}, current.data());
      return;
    }

    switch (precon.method()) {
    case precon_method::none:
      solver.iterate_solve(
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(num_entries, alpha, x, beta, y);
          disc.mpi_leader_apply_gpu(group_id{imex_implicit}, alpha * dt, x, 1, y);
        }, current, R);
    break;
    case precon_method::jacobi:
      solver.iterate_solve(
        [&](P y[]) -> void
        {
          tools::time_event timing_("jacobi preconditioner");
          gpu::jacobi_apply(precon.gpu_jacobi(), y);
        },
        [&](P alpha, P const x[], P beta, P y[]) -> void
        {
          gpu::axpby(num_entries, alpha, x, beta, y);
          disc.mpi_leader_apply_gpu(group_id{imex_implicit}, alpha * dt, x, 1, y);
        }, current, R);
    break;
    default:
      throw std::runtime_error("adi preconditioner not available for IMEX steppers");
    break;
    }

    disc.mpi_iteration_stop_gpu();
  }
}

template<typename P>
void imex_stepper<P>::next_step(
    discretization_manager<P> const &disc, gpu::vector<P> const &current,
    gpu::vector<P> &next) const
{
  tools::time_event performance_("stepper-imex-gpu");

  int64_t const num_entries = disc.num_dof();

  P const time = disc.time();
  P const dt   = disc.dt();

  gf.resize(num_entries);
  disc.ode_euler_gpu(group_id{imex_explicit}, time, current.data(), dt, gf.data());

  constexpr size_t stage0 = 0;
  implicit_solve(disc, stage0, time + dt, dt, precon1, gf, next);

  if (method == time_method::imex1)
    return;

  disc.ode_rhs_gpu(group_id{imex_explicit}, time + dt, next.data(), gf.data());

  if (disc.is_leader())
    // f = 0.5 * current + 0.5 * next + 0.5 * dt * f
    gpu::axpbygz(num_entries, 0.5, current.data(), 0.5, next.data(), 0.5 * dt, gf.data());

  constexpr size_t stage1 = 1;
  implicit_solve(disc, stage1, time + dt, P{0.5} * dt, precon2, gf, next);
}
#endif

template<typename P>
void imex_stepper<P>::print_bytes(std::ostream &os) const {
  os << "time-advance\n";
  os << "  workspace " << toMB((f.size()) * sizeof(P)) << '\n';
  os << "  solver    " << toMB(solver.used_bytes()) << '\n';
  os << "  precon    " << toMB(precon1.used_bytes() + precon2.used_bytes()) << '\n';
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
  assert(is_imex(data.step_method()));
  rassert(im.gid >= -1 and ex.gid >= -1,
          "the IMEX implicit and explicit groups have not been set in the pde_scheme");

  method = time_advance::imex_stepper<P>(options, im, ex);
}

template<typename P>
void time_advance_manager<P>::next_step(discretization_manager<P> const &dist,
                                        std::vector<P> const &current,
                                        std::vector<P> &next) const
{
  std::visit([&](auto const &stepper) {
                 stepper.next_step(dist, current, next);
             }, method);
}

#ifdef ASGARD_USE_GPU
template<typename P>
void time_advance_manager<P>::next_step(discretization_manager<P> const &dist,
                                        gpu::vector<P> const &current,
                                        gpu::vector<P> &next) const
{
  std::visit([&](auto const &stepper) {
                 stepper.next_step(dist, current, next);
             }, method);
}
#endif

template<typename P> // implemented in time-advance
bool advance_in_time(discretization_manager<P> &manager, int64_t num_steps)
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
    return true;

  sparse_grid &grid = manager.terms.grid;

  sparse_grid::strategy grid_strategy = sparse_grid::strategy::adapt;

  #ifdef ASGARD_USE_GPU
  // in GPU mode, move the data to the device and do not move back until the end
  // exception is made only when using refinement
  gpu::vector<P> &current = std::visit([&](auto &st) -> gpu::vector<P> & {
      return st.gcurrent;
    }, stepper.method);

  gpu::vector<P> &next = std::visit([&](auto &st) -> gpu::vector<P> & {
      return st.gnext;
    }, stepper.method);

  current = manager.state;

  auto found_bad = [&]() -> int { return gpu::num_non_finite(next); };

  auto resync_gpu = [&]() -> void {
      if (manager.is_leader())
        current.copy_to_host(manager.state);
      else
        manager.state.resize(manager.num_dof());
    };

  #else

  std::vector<P> &current = manager.state;
  std::vector<P> next;

  auto found_bad = [&]()
    -> size_t {
      size_t nbad = 0;
      #pragma omp parallel
      {
        size_t local_bad = 0;
        #pragma omp for
        for (size_t i = 0; i < next.size(); i++)
          if (std::abs(next[i]) > ASGARD_INFINITE_TRESHOLD)
            ++local_bad;

        #pragma omp atomic
        nbad += local_bad;
      }
      return nbad;
    };

  auto resync_gpu = [&]() -> void {};
  #endif

  auto accept_next = [&]() -> void {
      if (manager.is_leader())
        std::swap(next, current);
      else
        current.resize(manager.num_dof());
      params.take_step();
    };

  while (--num_steps >= 0)
  {
    stepper.next_step(manager, current, next);

    if (manager.safe_step) {
      tools::time_event performance_("check for inf/nan");

      if (found_bad() > 0) {
        std::cerr << "ERROR: found 'inf' or 'nan' entries in the next time-step\n"
                  << "       this is an indication of either bad pde_scheme or incompatible ASGarD options\n"
                  << "       e.g., adaptive tolerance or solver tolerance is too high,\n"
                  << "       max-grid level is too low, time-step is too large, etc.\n";
        resync_gpu();
        return false;
      }
    }

    if (manager.refinement)
    {
      int const gen = grid.generation();
      manager.refine(grid_strategy, next);
      manager.grid_sync(); // no-op, unless MPI or GPUs are enabled

      if (grid.generation() != gen) {
        manager.update_grid(next);
        if (stepper.is_steady_state()) {
          num_steps = 1;
          grid_strategy = sparse_grid::strategy::refine;
        }
      }
    }

    accept_next();

    if (not manager.stop_verbosity()) {
      // if verbosity is not turned off, report every 2 or 10 seconds
      double duration = tools::simple_timer::duration_since(wctime);
      if ((manager.high_verbosity() and duration > 2000) or (duration > 10000)) {
        manager.progress_report(std::cout, static_cast<int64_t>(current.size()));
        wctime = tools::simple_timer::current_time();
        if (manager.options_.show_memusage)
          manager.report_memusage();
      }
    }

    if (stepper.is_steady_state())
      params.set_final_time();
  }
  resync_gpu();

  return true;
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct time_advance::steady_state<double>;
template struct time_advance::rungekutta<double>;
template struct time_advance::crank_nicolson<double>;
template struct time_advance_manager<double>;

template bool advance_in_time(discretization_manager<double> &, int64_t);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct time_advance::steady_state<float>;
template struct time_advance::rungekutta<float>;
template struct time_advance::crank_nicolson<float>;
template struct time_advance_manager<float>;

template bool advance_in_time(discretization_manager<float> &, int64_t);
#endif
} // namespace asgard
