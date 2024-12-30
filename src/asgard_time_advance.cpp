#include "asgard_time_advance.hpp"

namespace asgard::time_advance
{
template<typename P>
fk::vector<P>
rungekutta3(discretization_manager<P> const &dist, std::vector<P> const &current)
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

  solve_opts const solver = options.solver.value();

  static std::vector<P> rhs;
  disc.ode_irhs(disc.time() + dt, current, rhs);

  std::optional<matrix_factor<P>> &euler_mat = disc.get_op_matrix();

  // if using a direct solver, on the first run, we need to update the matrices
  if (solver == solve_opts::direct and not euler_mat)
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

  if (solver == solve_opts::direct)
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
  solve_opts solver  = options.solver.value();
  P const tolerance  = *options.isolver_tolerance;
  int const restart  = *options.isolver_iterations;
  int const max_iter = *options.isolver_outer_iterations;
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
    if (solver == solve_opts::gmres)
    {
      pde.gmres_outputs[0] = solver::simple_gmres_euler(
          pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_1, f, restart, max_iter, tolerance);
    }
    else if (solver == solve_opts::bicgstab)
    {
      pde.gmres_outputs[0] = solver::bicgstab_euler(
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

    if (solver == solve_opts::gmres)
    {
      pde.gmres_outputs[1] = solver::simple_gmres_euler(
          P{0.5} * pde.get_dt(), imex_flag::imex_implicit, operator_matrices,
          f_2, f, restart, max_iter, tolerance);
    }
    else if (solver == solve_opts::bicgstab)
    {
      pde.gmres_outputs[1] = solver::bicgstab_euler(
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
void advance_time(discretization_manager<P> &manager, int64_t num_steps)
{
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
        case time_advance::method::exp:
          return time_advance::rungekutta3(manager, manager.current_state());
        case time_advance::method::imp:
          return time_advance::implicit_advance<P>(manager, manager.current_state());
        case time_advance::method::imex:
          return time_advance::imex_advance<P>(manager, pde, kronops, grid,
                                               manager.current_state(), fk::vector<P>(),
                                               time);
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
          case time_advance::method::exp:
            return time_advance::rungekutta3(manager, y.to_std());
          case time_advance::method::imp:
            return time_advance::implicit_advance<P>(manager, y.to_std());
          case time_advance::method::imex:
            return time_advance::imex_advance<P>(manager, pde, kronops, grid,
                                                 y, y_first_refine, time);
          default:
            return fk::vector<P>();
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

#ifdef ASGARD_ENABLE_DOUBLE
template void advance_time(discretization_manager<double> &, int64_t);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void advance_time(discretization_manager<float> &, int64_t);
#endif
} // namespace asgard
