#include "asgard_discretization.hpp"

#include "asgard_small_mats.hpp"

namespace asgard::time_advance
{
template<typename P>
fk::vector<P>
rungekutta3_m(discretization_manager<P> const &dist, std::vector<P> const &current)
{
  P const dt = dist.dt();

  // 3 right-hand-sides and the intermediate step
  // the assumption is that the time-stepping scheme does not change much
  // thus it makes sense to make these static and avoid repeated allocation
  static std::vector<P> k1, k2, k3, s1;

  k1.resize(current.size());
  k2.resize(current.size());
  k3.resize(current.size());
  s1.resize(current.size());

  dist.ode_rhs(imex_flag::unspecified, dist.time(), current, k1);

  for (auto i : indexof(s1))
    s1[i] = current[i] + 0.5 * dt * k1[i];

  dist.ode_rhs(imex_flag::unspecified, dist.time() + 0.5 * dt, s1, k2);

  for (auto i : indexof(s1))
    s1[i] = current[i] - dt * k1[i] + 2 * dt * k2[i];

  dist.ode_rhs(imex_flag::unspecified, dist.time() + dt, s1, k3);

  fk::vector<P> r(current.size());
  for (auto i : indexof(r))
    r[i] = current[i] + dt * (k1[i] + 4 * k2[i] + k3[i]) / P{6};

  return r;
}

// no-MPI solver yet
template<typename P>
fk::vector<P>
implicit_advance(discretization_manager<P> const &disc, std::vector<P> const &current)
{
  P const dt = disc.dt();

  auto const &options = disc.get_pde().options();

  solver_method const solver = options.solver.value();

  static std::vector<P> rhs;
  disc.ode_irhs(disc.time() + dt, current, rhs);

  std::optional<matrix_factor<P>> &euler_mat = disc.get_op_matrix();

  // if using a direct solver, on the first run, we need to update the matrices
  if (solver == solver_method::direct and not euler_mat)
  {
    auto const &table   = disc.get_grid().get_table();
    auto const &subgrid = disc.get_grid().get_subgrid(get_rank());

    int const rows = disc.get_hiermanip().block_size() * subgrid.nrows();
    int const cols = disc.get_hiermanip().block_size() * subgrid.ncols();

    // must form the matrix
    matrix_factor<P> euler;

    euler.A.clear_and_resize(rows, cols);
    build_system_matrix<P>(
        disc.get_pde(), [&](int t, int d)->fk::matrix<P>{ return disc.get_coeff_matrix(t, d); },
        table, euler.A, subgrid);

    // AA = I - dt*A;
    fm::scal(-dt, euler.A);
    if (subgrid.row_start == subgrid.col_start)
    {
      for (int i = 0; i < euler.A.nrows(); ++i)
      {
        euler.A(i, i) += 1.0;
      }
    }

    euler.ipiv.resize(euler.A.nrows());
    // one-shot factorize and solve
    fm::gesv(euler.A, rhs, euler.ipiv);
    euler_mat = std::move(euler);
    return rhs;
  } // end first time/update system

  if (solver == solver_method::direct)
  { // reusing the computed factor
    fm::getrs(euler_mat->A, rhs, euler_mat->ipiv);
    return rhs;
  }
  else
  {
    disc.ode_sv(imex_flag::unspecified, rhs);
    return rhs;
  }
}

// this function executes an implicit-explicit (imex) time step using the
// current solution vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
imex_advance(discretization_manager<P> &disc,
             PDE<P> &pde, kron_operators<P> &operator_matrices,
             adapt::distributed_grid<P> const &adaptive_grid,
             fk::vector<P> const &f_0, fk::vector<P> const &x_prev,
             P const time)
{
  // BEFE = 0 case
  expect(time >= 0);

  auto const &options = pde.options();

  P const dt       = pde.get_dt();

#ifdef ASGARD_USE_CUDA
  fk::vector<P, mem_type::owner, imex_resrc> f = f_0.clone_onto_device();
  fk::vector<P, mem_type::owner, imex_resrc> f_orig_dev =
      f_0.clone_onto_device();
#else
  fk::vector<P, mem_type::owner, imex_resrc> f          = f_0;
  fk::vector<P, mem_type::owner, imex_resrc> f_orig_dev = f_0;

  int const degree = pde.get_dimensions()[0].get_degree();

  auto const &plan       = adaptive_grid.get_distrib_plan();
  auto const &grid       = adaptive_grid.get_subgrid(get_rank());
  int const elem_size    = fm::ipow(degree + 1, pde.num_dims());
  int const A_local_rows = elem_size * grid.nrows();

  fk::vector<P, mem_type::owner, imex_resrc> reduced_fx(A_local_rows);
#endif

#ifdef ASGARD_USE_CUDA
  disc.do_poisson_update(f.clone_onto_host().to_std());
#else
  disc.do_poisson_update(f.to_std());
#endif

  disc.compute_coefficients(coeff_update_mode::imex_explicit);

  operator_matrices.reset_coefficients(imex_flag::imex_explicit, pde,
                                       disc.get_cmatrices(), adaptive_grid);

  // Explicit step f_1s = f_0 + dt A f_0
  fk::vector<P, mem_type::owner, imex_resrc> fx(f.size());

  tools::timer.start("explicit_1");
  {
    tools::time_event kronm_(
        "kronmult - explicit 1", operator_matrices.flops(imex_flag::imex_explicit));
    operator_matrices.template apply<imex_resrc>(imex_flag::imex_explicit, 1.0, f.data(), 0.0, fx.data());
  }

#ifndef ASGARD_USE_CUDA
  reduce_results(fx, reduced_fx, plan, get_rank());

  exchange_results(reduced_fx, fx, elem_size, plan, get_rank());
  fm::axpy(fx, f, dt); // f here is f_1s
#else
  fm::axpy(fx, f, dt);   // f here is f_1s
#endif

  tools::timer.stop("explicit_1");

  // Implicit step f_1: f_1 - dt B f_1 = f_1s
  solver_method solver  = options.solver.value();
  P const tolerance  = *options.isolver_tolerance;
  int const restart  = *options.isolver_inner_iterations;
  int const max_iter = *options.isolver_iterations;
  fk::vector<P, mem_type::owner, imex_resrc> f_1(f.size());
  fk::vector<P, mem_type::owner, imex_resrc> f_1_output(f.size());
  if (pde.do_collision_operator())
  {
    tools::timer.start("implicit_1");

#ifdef ASGARD_USE_CUDA
    disc.compute_moments(f.clone_onto_host().to_std());
#else
    disc.compute_moments(f.to_std());
#endif
    disc.compute_coefficients(coeff_update_mode::imex_implicit);

    // f2 now
    operator_matrices.reset_coefficients(imex_flag::imex_implicit, pde,
                                         disc.get_cmatrices(), adaptive_grid);

    // use previous refined solution as initial guess to GMRES if it exists
    if (x_prev.empty())
    {
      f_1 = f; // use f_1s as input
    }
    else
    {
      if constexpr (imex_resrc == resource::device)
      {
        f_1 = x_prev.clone_onto_device();
      }
      else
      {
        f_1 = x_prev;
      }
    }
    if (solver == solver_method::gmres)
    {
      pde.gmres_outputs[0] = solvers::simple_gmres_euler(
          pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_1, f, restart, max_iter, tolerance);
    }
    else if (solver == solver_method::bicgstab)
    {
      pde.gmres_outputs[0] = solvers::bicgstab_euler(
          pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_1, f, max_iter, tolerance);
    }
    else
    {
      throw std::runtime_error("imex solver must be gmres or bicgstab.");
    }
    // save output of GMRES call to use in the second one
    f_1_output = f_1;

    tools::timer.stop("implicit_1");
  }
  else
  {
    // for non-collision: f_1 = f_1s
    fm::copy(f, f_1);
  }

  // --------------------------------
  // Second Stage
  // --------------------------------
  fm::copy(f_orig_dev, f); // f here is now f_0

#ifdef ASGARD_USE_CUDA
  disc.do_poisson_update(f_1.clone_onto_host().to_std());
#else
  disc.do_poisson_update(f_1.to_std());
#endif

  disc.compute_coefficients(coeff_update_mode::imex_explicit);

  operator_matrices.reset_coefficients(imex_flag::imex_explicit, pde,
                                       disc.get_cmatrices(), adaptive_grid);

  tools::timer.start("explicit_2");
  // Explicit step f_2s = 0.5*f_0 + 0.5*(f_1 + dt A f_1)
  {
    tools::time_event kronm_(
        "kronmult - explicit 2", operator_matrices.flops(imex_flag::imex_explicit));
    operator_matrices.template apply<imex_resrc>(imex_flag::imex_explicit, 1.0, f_1.data(), 0.0, fx.data());
  }

#ifndef ASGARD_USE_CUDA
  reduce_results(fx, reduced_fx, plan, get_rank());

  // fk::vector<P, mem_type::owner, resource::host> t_f2(x_orig.size());
  exchange_results(reduced_fx, fx, elem_size, plan, get_rank());
  fm::axpy(fx, f_1, dt); // f_1 here is now f_2 = f_1 + dt*T(f_1)
#else
  fm::axpy(fx, f_1, dt); // f_1 here is now f_2 = f_1 + dt*T(f_1)
#endif

  fm::axpy(f_1, f);    // f is now f_0 + f_2
  fm::scal(P{0.5}, f); // f = 0.5 * (f_0 + f_2) = f_2s
  tools::timer.stop("explicit_2");

  // Implicit step f_2: f_2 - dt B f_2 = f_2s
  if (pde.do_collision_operator())
  {
    tools::timer.start("implicit_2");

    // Update coeffs
#ifdef ASGARD_USE_CUDA
    disc.compute_moments(f.clone_onto_host().to_std());
#else
    disc.compute_moments(f.to_std());
#endif
    disc.compute_coefficients(coeff_update_mode::imex_implicit);

    tools::timer.start("implicit_2_solve");
    fk::vector<P, mem_type::owner, imex_resrc> f_2(f.size());
    if (x_prev.empty())
    {
      f_2 = std::move(f_1_output);
    }
    else
    {
      if constexpr (imex_resrc == resource::device)
      {
        f_2 = x_prev.clone_onto_device();
      }
      else
      {
        f_2 = x_prev;
      }
    }

    operator_matrices.reset_coefficients(imex_flag::imex_implicit, pde,
                                         disc.get_cmatrices(), adaptive_grid);

    if (solver == solver_method::gmres)
    {
      pde.gmres_outputs[1] = solvers::simple_gmres_euler(
          P{0.5} * pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_2, f, restart, max_iter, tolerance);
    }
    else if (solver == solver_method::bicgstab)
    {
      pde.gmres_outputs[1] = solvers::bicgstab_euler(
          P{0.5} * pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_2, f, max_iter, tolerance);
    }
    else
    {
      throw std::runtime_error("imex solver must be gmres or bicgstab.");
    }
    tools::timer.stop("implicit_2_solve");
    tools::timer.stop("implicit_2");
    if constexpr (imex_resrc == resource::device)
    {
      return f_2.clone_onto_host();
    }
    else
    {
      return f_2;
    }
  }
  else
  {
    // for non-collision: f_2 = f_2s, and f here is f_2s
    if constexpr (imex_resrc == resource::device)
    {
      return f.clone_onto_host();
    }
    else
    {
      return f;
    }
  }
}

} // namespace asgard::time_advance

namespace asgard
{
template<typename P> // implemented in time-advance
void advance_in_time(discretization_manager<P> &manager, int64_t num_steps)
{
  if (manager.version2()) {
    advance_time_v2(manager, num_steps);
    return;
  }

  if (num_steps == 0)
    return;
  num_steps = std::max(int64_t{-1}, num_steps);

  auto &pde  = *manager.pde;
  auto &grid = manager.grid;

  auto &kronops     = manager.kronops;

  auto const method = pde.options().step_method.value();

  if (manager.high_verbosity())
    node_out() << "--- begin time loop w/ dt " << pde.get_dt() << " ---\n";

  while (manager.time_step_ < manager.final_time_step_)
  {
    // take a time advance step
    auto const time           = manager.time();
    const std::string time_id = tools::timer.start("time_advance");

    fk::vector<P> f_val = [&]()
        -> fk::vector<P> {
      if (not pde.options().adapt_threshold)
      {
        switch (method)
        {
        case time_method::exp:
          return time_advance::rungekutta3_m(manager, manager.current_state());
        case time_method::imp:
          return time_advance::implicit_advance<P>(manager, manager.current_state());
        case time_method::imex:
          return time_advance::imex_advance<P>(manager, pde, kronops, grid,
                                               manager.current_state(), fk::vector<P>(),
                                               time);
        default:
          throw std::runtime_error("old time-advance called with new enum-tag");
        };
      }

      // coarsen
      auto old_size = grid.size();
      auto y        = grid.coarsen_solution(pde, manager.current_state());
      if (manager.high_verbosity())
        node_out() << " adapt -- coarsened grid from " << old_size << " -> "
                   << grid.size() << " elems\n";

      // clear the pre-computed components if the coarsening removed indexes
      if (old_size != grid.size())
        manager.update_grid_components();

      // save coarsen stats
      pde.adapt_info.initial_dof = old_size;
      pde.adapt_info.coarsen_dof = grid.size();
      pde.adapt_info.refine_dofs = std::vector<int>();
      // save GMRES stats starting with the coarsen stats
      pde.adapt_info.gmres_stats =
          std::vector<std::vector<gmres_info<P>>>({pde.gmres_outputs});

      // refine
      bool refining = true;
      fk::vector<P> y_first_refine;
      while (refining)
      {
        // take a probing refinement step
        fk::vector<P> y_stepped = [&]() {
          switch (method)
          {
          case time_method::exp:
            return time_advance::rungekutta3_m(manager, y.to_std());
          case time_method::imp:
            return time_advance::implicit_advance<P>(manager, y.to_std());
          case time_method::imex:
            return time_advance::imex_advance<P>(manager, pde, kronops, grid,
                                                 y, y_first_refine, time);
          default:
            throw std::runtime_error("old time-advance called with new enum-tag");
          };
        }();

        auto const old_plan = grid.get_distrib_plan();
        old_size            = grid.size();

        fk::vector<P> y_refined = grid.refine_solution(pde, y_stepped);
        // if either one of the ranks reports 1, i.e., y_stepped.size() changed
        refining = get_global_max<bool>(y_stepped.size() != y_refined.size(),
                                        grid.get_distrib_plan());

        if (manager.high_verbosity())
          node_out() << " adapt -- refined grid from " << old_size << " -> "
                     << grid.size() << " elems\n";
        // save refined DOF stats
        pde.adapt_info.refine_dofs.push_back(grid.size());
        // append GMRES stats for refinement
        pde.adapt_info.gmres_stats.push_back({pde.gmres_outputs});

        if (!refining)
        {
          y = std::move(y_stepped);
        }
        else
        {
          // added more indexes, matrices will have to be remade
          manager.update_grid_components();

          y = grid.redistribute_solution(y, old_plan, old_size);

          // after first refinement, save the refined vector to use as initial
          // "guess" to GMRES
          if (y_first_refine.empty())
          {
            y_first_refine = std::move(y_refined);
          }

          // pad with zeros if more elements were added
          y_first_refine.resize(y.size());
        }
      }

      return y;
    }();

    tools::timer.stop(time_id);

    // advances time and the time-step
    manager.set_next_step(f_val);

    if (manager.high_verbosity() and not pde.options().ignore_exact)
    {
      auto session = tools::time_session("compute exact solution");
      auto rmse = manager.rmse_exact_sol();
      if (rmse)
      {
        auto const &rmse_errors     = rmse.value()[0];
        auto const &relative_errors = rmse.value()[1];
        expect(rmse_errors.size() == relative_errors.size());
        for (auto j : indexof(rmse_errors))
        {
          node_out() << "Errors for local rank: " << j << '\n';
          node_out() << "RMSE (numeric-analytic) [wavelet]: "
                     << rmse_errors[j] << '\n';
          node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                     << relative_errors[j] << " %" << '\n';
        }
      }

      node_out() << "complete timestep: " << manager.time_step_ << '\n';
    }

    if (--num_steps == 0)
      break;
  }
}
}

namespace asgard::time_advance
{

template<typename P>
void steady_state<P>::next_step(
    discretization_manager<P> const &disc, std::vector<P> const &current,
    std::vector<P> &endstep) const
{
  tools::time_event performance_("solve steady state");

  P const time = disc.time_params().stop_time();

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  if (solver.grid_gen != disc.get_sgrid().generation())
    solver.update_grid(disc.get_sgrid(), disc.get_conn(), disc.get_terms(), 0);

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

  P const time = disc.time_params().time();
  P const dt   = disc.time_params().dt();

  switch (rktype) {
    case time_method::forward_euler:
      k1.resize(current.size());
      disc.ode_rhs_v2(time, current, k1);

      next.resize(current.size());

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        next[i] = current[i] + dt * k1[i];
      break;
    case time_method::rk2:
      k1.resize(current.size());
      k2.resize(current.size());
      s1.resize(current.size());

      disc.ode_rhs_v2(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs_v2(time + 0.5 * dt, s1, k2);

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

      disc.ode_rhs_v2(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs_v2(time + 0.5 * dt, s1, k2);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] - dt * k1[i] + 2 * dt * k2[i];

      disc.ode_rhs_v2(time + dt, s1, k3);

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

      disc.ode_rhs_v2(time, current, k1);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k1[i];

      disc.ode_rhs_v2(time + 0.5 * dt, s1, k2);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + 0.5 * dt * k2[i];

      disc.ode_rhs_v2(time + 0.5 * dt, s1, k3);

      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < current.size(); i++)
        s1[i] = current[i] + dt * k3[i];

      disc.ode_rhs_v2(time + dt, s1, k4);

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

  P const time = disc.time_params().time();
  P const dt   = disc.time_params().dt();

  P const substep = (method == time_method::cn) ? 0.5 : 1;

  // if the grid changed since the last time we used the solver
  // update the matrices and preconditioners, update-grid checks what's needed
  if (solver.grid_gen != disc.get_sgrid().generation())
    solver.update_grid(disc.get_sgrid(), disc.get_conn(), disc.get_terms(), substep * dt);

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

}

namespace asgard
{

template<typename P>
time_advance_manager<P>::time_advance_manager(time_data<P> const &tdata, prog_opts const &options)
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
    default:
      throw std::runtime_error("unimplemented time-advance option");
  };
}

template<typename P>
std::string time_advance_manager<P>::method_name() const {
  std::map<time_method, std::string> names = {
    {time_method::steady, "Steady state solver"},
    {time_method::forward_euler, "Forward-Euler 1-step (explicit)"},
    {time_method::rk2, "Runge-Kutta 2-step (explicit)"},
    {time_method::rk3, "Runge-Kutta 3-step (explicit)"},
    {time_method::rk4, "Runge-Kutta 4-step (explicit)"},
    {time_method::cn, "Crank-Nicolson 1-step (implicit)"},
    {time_method::back_euler, "Backward-Euler 1-step (implicit)"},
  };

  return names.find(data.step_method())->second;
}

template<typename P> // implemented in time-advance
void advance_time_v2(discretization_manager<P> &manager, int64_t num_steps)
{
  // periodically reports time
  static tools::simple_timer::time_point wctime = tools::simple_timer::current_time();

  time_advance_manager<P> const &stepper = manager.stepper;

  time_data<P> &params = manager.stepper.data;

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

  P const tol = manager.get_pde2().options().adapt_threshold.value_or(-1);

  sparse_grid &grid = manager.sgrid;

  sparse_grid::strategy grid_strategy = sparse_grid::strategy::refine;

  std::vector<P> next;
  while (--num_steps >= 0)
  {
    stepper.next_step(manager, manager.state, next);

    if (tol > 0) {
      int const gen = grid.generation();
      grid.refine(tol, manager.hier.block_size(), manager.conn[connect_1d::hierarchy::volume],
                  grid_strategy, next);
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
template void advance_time_v2(discretization_manager<double> &, int64_t);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct time_advance::steady_state<float>;
template struct time_advance::rungekutta<float>;
template struct time_advance::crank_nicolson<float>;
template struct time_advance_manager<float>;

template void advance_in_time(discretization_manager<float> &, int64_t);
template void advance_time_v2(discretization_manager<float> &, int64_t);
#endif
} // namespace asgard
