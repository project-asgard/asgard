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
             PDE<P> &pde, std::vector<moment<P>> &moments,
             kron_operators<P> &operator_matrices,
             adapt::distributed_grid<P> const &adaptive_grid,
             basis::wavelet_transform<P, resource::host> const &transformer,
             fk::vector<P> const &f_0, fk::vector<P> const &x_prev,
             P const time)
{
  // BEFE = 0 case
  expect(time >= 0);
  expect(moments.size() > 0);

  auto const &options = pde.options();

  // create 1D version of PDE and element table for wavelet->realspace mappings
  PDE pde_1d       = PDE(pde, PDE<P>::extract_dim0);
  int const degree = pde.get_dimensions()[0].get_degree();
  int const level  = pde.get_dimensions()[0].get_level();

  adapt::distributed_grid adaptive_grid_1d(pde_1d);

  // Create workspace for wavelet transform
  int const dense_size      = dense_space_size(pde_1d);
  int const quad_dense_size = dense_dim_size(
      ASGARD_NUM_QUADRATURE - 1, pde_1d.get_dimensions()[0].get_level());
  fk::vector<P, mem_type::owner, resource::host> workspace(quad_dense_size * 2);
  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace, 0,
                                                    quad_dense_size - 1),
      fk::vector<P, mem_type::view, resource::host>(workspace, quad_dense_size,
                                                    quad_dense_size * 2 - 1)};

  auto const dt        = pde.get_dt();
  P const min          = pde.get_dimensions()[0].domain_min;
  P const max          = pde.get_dimensions()[0].domain_max;
  int const N_elements = fm::ipow2(level);

  auto nodes = gen_realspace_nodes(degree, level, min, max);

#ifdef ASGARD_USE_CUDA
  fk::vector<P, mem_type::owner, imex_resrc> f = f_0.clone_onto_device();
  fk::vector<P, mem_type::owner, imex_resrc> f_orig_dev =
      f_0.clone_onto_device();
#else
  fk::vector<P, mem_type::owner, imex_resrc> f          = f_0;
  fk::vector<P, mem_type::owner, imex_resrc> f_orig_dev = f_0;

  auto const &plan       = adaptive_grid.get_distrib_plan();
  auto const &grid       = adaptive_grid.get_subgrid(get_rank());
  int const elem_size    = fm::ipow(degree + 1, pde.num_dims());
  int const A_local_rows = elem_size * grid.nrows();

  fk::vector<P, mem_type::owner, imex_resrc> reduced_fx(A_local_rows);
#endif

  auto do_poisson_update = [&](fk::vector<P, mem_type::owner, imex_resrc> const
                                   &f_in) {
    fk::vector<P> poisson_source(quad_dense_size);
    fk::vector<P> phi(quad_dense_size);
    fk::vector<P> poisson_E(quad_dense_size);
    {
      tools::time_event pupdate_("poisson_update");
      // Get 0th moment
      fk::vector<P, mem_type::owner, imex_resrc> mom0(dense_size);
      fm::sparse_gemv(moments[0].get_moment_matrix_dev(), f_in, mom0);
      fk::vector<P> &mom0_real = moments[0].create_realspace_moment(
          pde_1d, mom0, adaptive_grid_1d.get_table(), transformer,
          tmp_workspace);
      param_manager.get_parameter("n")->value = [&](P const x_v,
                                                    P const t = 0) -> P {
        ignore(t);
        return interp1(nodes, mom0_real, {x_v})[0];
      };

      // Compute source for poisson
      std::transform(mom0_real.begin(), mom0_real.end(), poisson_source.begin(),
                     [](P const &x_v) {
                       return param_manager.get_parameter("S")->value(x_v, 0.0);
                     });

      solver::poisson_solver(poisson_source, pde.poisson_diag,
                             pde.poisson_off_diag, phi, poisson_E,
                             ASGARD_NUM_QUADRATURE - 2, N_elements, min, max,
                             static_cast<P>(0.0), static_cast<P>(0.0),
                             solver::poisson_bc::periodic);

      param_manager.get_parameter("E")->value =
          [poisson_E, nodes](P const x_v, P const t = 0) -> P {
        ignore(t);
        return interp1(nodes, poisson_E, {x_v})[0];
      };

      pde.E_field  = poisson_E;
      pde.E_source = poisson_source;
      pde.phi      = phi;

      P const max_E = std::abs(*std::max_element(
          poisson_E.begin(), poisson_E.end(), [](const P &x_v, const P &y_v) {
            return std::abs(x_v) < std::abs(y_v);
          }));

      param_manager.get_parameter("MaxAbsE")->value =
          [max_E](P const x_v, P const t = 0) -> P {
        ignore(t);
        ignore(x_v);
        return max_E;
      };
    }

    disc.compute_coefficients();
  };

  auto calculate_moments =
      [&](fk::vector<P, mem_type::owner, imex_resrc> const &f_in) {
        // \int f dv
        fk::vector<P, mem_type::owner, imex_resrc> mom0(dense_size);
        fm::sparse_gemv(moments[0].get_moment_matrix_dev(), f_in, mom0);
        fk::vector<P> &mom0_real = moments[0].create_realspace_moment(
            pde_1d, mom0, adaptive_grid_1d.get_table(), transformer,
            tmp_workspace);
        // n = \int f dv
        param_manager.get_parameter("n")->value = [&](P const x_v,
                                                      P const t = 0) -> P {
          ignore(t);
          return interp1(nodes, mom0_real, {x_v})[0];
        };

        // \int f v_x dv
        fk::vector<P, mem_type::owner, imex_resrc> mom1(dense_size);
        fm::sparse_gemv(moments[1].get_moment_matrix_dev(), f_in, mom1);
        fk::vector<P> &mom1_real = moments[1].create_realspace_moment(
            pde_1d, mom1, adaptive_grid_1d.get_table(), transformer,
            tmp_workspace);

        // u_x = \int f v_x  dv / n
        param_manager.get_parameter("u")->value = [&](P const x_v,
                                                      P const t = 0) -> P {
          return interp1(nodes, mom1_real, {x_v})[0] /
                 param_manager.get_parameter("n")->value(x_v, t);
        };
        if (pde.num_dims() == 3 && moments.size() > 3)
        {
          // Calculate additional moments for PDEs with more than one velocity
          // dimension

          // \int f v_{y} dv
          fk::vector<P, mem_type::owner, imex_resrc> mom2(dense_size);
          fm::sparse_gemv(moments[2].get_moment_matrix_dev(), f_in, mom2);
          fk::vector<P> &mom2_real = moments[2].create_realspace_moment(
              pde_1d, mom2, adaptive_grid_1d.get_table(), transformer,
              tmp_workspace);

          // u_y = \int_v f v_y dv / n
          param_manager.get_parameter("u2")->value = [&](P const x_v,
                                                         P const t = 0) -> P {
            return interp1(nodes, mom2_real, {x_v})[0] /
                   param_manager.get_parameter("n")->value(x_v, t);
          };

          // \int f v_x^2 dv
          fk::vector<P, mem_type::owner, imex_resrc> mom3(dense_size);
          fm::sparse_gemv(moments[3].get_moment_matrix_dev(), f_in, mom3);
          fk::vector<P> &mom3_real = moments[3].create_realspace_moment(
              pde_1d, mom3, adaptive_grid_1d.get_table(), transformer,
              tmp_workspace);

          // \int f v_y^2 dv
          fk::vector<P, mem_type::owner, imex_resrc> mom4(dense_size);
          fm::sparse_gemv(moments[4].get_moment_matrix_dev(), f_in, mom4);
          fk::vector<P> &mom4_real = moments[4].create_realspace_moment(
              pde_1d, mom4, adaptive_grid_1d.get_table(), transformer,
              tmp_workspace);

          // \theta = \frac{ \int f(v_x^2 + v_y^2) dv }{2n} - 0.5 * (u_x^2
          // + u_y^2)
          param_manager.get_parameter("theta")->value =
              [&](P const x_v, P const t = 0) -> P {
            P const mom3_x = interp1(nodes, mom3_real, {x_v})[0];
            P const mom4_x = interp1(nodes, mom4_real, {x_v})[0];

            P const u1 = param_manager.get_parameter("u")->value(x_v, t);
            P const u2 = param_manager.get_parameter("u2")->value(x_v, t);

            P const n = param_manager.get_parameter("n")->value(x_v, t);

            return (mom3_x + mom4_x) / (2.0 * n) -
                   0.5 * (std::pow(u1, 2) + std::pow(u2, 2));
          };
        }
        else if (pde.num_dims() == 4 && moments.size() > 6)
        {
          // Moments for 1X3V case
          // TODO: this will be refactored to replace dimension cases in the
          // future
          std::vector<fk::vector<P, mem_type::owner, imex_resrc>> vec_moments;
          std::vector<fk::vector<P> *> moments_real;
          // Create moment matrices and realspace moments for all moments in PDE
          for (size_t mom = 2; mom < moments.size(); mom++)
          {
            // \int f v_x dv
            vec_moments.push_back(
                fk::vector<P, mem_type::owner, imex_resrc>(dense_size));
            fm::sparse_gemv(moments[mom].get_moment_matrix_dev(), f_in,
                            vec_moments.back());
            moments_real.push_back(&moments[mom].create_realspace_moment(
                pde_1d, vec_moments.back(), adaptive_grid_1d.get_table(),
                transformer, tmp_workspace));
          }

          // u_y = \int_v f v_y dv / n
          param_manager.get_parameter("u2")->value =
              [&nodes, moments_real](P const x_v, P const t = 0) -> P {
            return interp1(nodes, *(moments_real[0]), {x_v})[0] /
                   param_manager.get_parameter("n")->value(x_v, t);
          };

          // u_z = \int_v f v_z dv / n
          param_manager.get_parameter("u3")->value =
              [&nodes, moments_real](P const x_v, P const t = 0) -> P {
            return interp1(nodes, *(moments_real[1]), {x_v})[0] /
                   param_manager.get_parameter("n")->value(x_v, t);
          };

          // \theta = \frac{ \int f(v_x^2 + v_y^2 + v_z^2) dv }{ 3n }
          //          - (1/3) * (u_x^2 + u_y^2 + u_z^2)
          param_manager.get_parameter("theta")->value =
              [&nodes, moments_real](P const x_v, P const t = 0) -> P {
            P const mom4_x = interp1(nodes, *(moments_real[2]), {x_v})[0];
            P const mom5_x = interp1(nodes, *(moments_real[3]), {x_v})[0];
            P const mom6_x = interp1(nodes, *(moments_real[4]), {x_v})[0];

            P const u1 = param_manager.get_parameter("u")->value(x_v, t);
            P const u2 = param_manager.get_parameter("u2")->value(x_v, t);
            P const u3 = param_manager.get_parameter("u3")->value(x_v, t);

            P const n = param_manager.get_parameter("n")->value(x_v, t);

            return (mom4_x + mom5_x + mom6_x) / (3.0 * n) -
                   (1.0 / 3.0) * (u1 * u1 + u2 * u2 + u3 * u3);
          };
        }
        else
        {
          // theta moment for 1x1v case
          fk::vector<P, mem_type::owner, imex_resrc> mom2(dense_size);
          fm::sparse_gemv(moments[2].get_moment_matrix_dev(), f_in, mom2);
          fk::vector<P> &mom2_real = moments[2].create_realspace_moment(
              pde_1d, mom2, adaptive_grid_1d.get_table(), transformer,
              tmp_workspace);
          // \theta = \int f v_x^2 dv / n - u_x^2
          param_manager.get_parameter("theta")->value =
              [&](P const x_v, P const t = 0) -> P {
            P const u = param_manager.get_parameter("u")->value(x_v, t);
            return (interp1(nodes, mom2_real, {x_v})[0] /
                    param_manager.get_parameter("n")->value(x_v, t)) -
                   std::pow(u, 2);
          };
        }
      };

  if (pde.do_poisson_solve())
  {
    do_poisson_update(f);
  }

  operator_matrices.reset_coefficients(imex_flag::imex_explicit, pde,
                                       disc.get_cmatrices(), adaptive_grid);

  // Explicit step f_1s = f_0 + dt A f_0
  tools::timer.start("explicit_1");
  fk::vector<P, mem_type::owner, imex_resrc> fx(f.size());

  {
    tools::time_event kronm_(
        "kronmult - explicit", operator_matrices.flops(imex_flag::imex_explicit));
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
  tools::timer.start("implicit_1");

  // Create rho_1s
  calculate_moments(f);

  // Implicit step f_1: f_1 - dt B f_1 = f_1s
  solve_opts solver  = options.solver.value();
  P const tolerance  = *options.isolver_tolerance;
  int const restart  = *options.isolver_iterations;
  int const max_iter = *options.isolver_outer_iterations;
  fk::vector<P, mem_type::owner, imex_resrc> f_1(f.size());
  fk::vector<P, mem_type::owner, imex_resrc> f_1_output(f.size());
  if (pde.do_collision_operator())
  {
    disc.compute_coefficients();

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
  }
  else
  {
    // for non-collision: f_1 = f_1s
    fm::copy(f, f_1);
  }

  tools::timer.stop("implicit_1");

  // --------------------------------
  // Second Stage
  // --------------------------------
  tools::timer.start("explicit_2");
  fm::copy(f_orig_dev, f); // f here is now f_0

  if (pde.do_poisson_solve())
  {
    do_poisson_update(f_1);
  }

  operator_matrices.reset_coefficients(imex_flag::imex_explicit, pde,
                                       disc.get_cmatrices(), adaptive_grid);

  // Explicit step f_2s = 0.5*f_0 + 0.5*(f_1 + dt A f_1)
  {
    tools::time_event kronm_(
        "kronmult - explicit", operator_matrices.flops(imex_flag::imex_explicit));
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
  if (pde.do_collision_operator())
  {
    tools::timer.start("implicit_2");
  }
  tools::timer.start("implicit_2_mom");
  // Create rho_2s
  calculate_moments(f);
  tools::timer.stop("implicit_2_mom");

  // Implicit step f_2: f_2 - dt B f_2 = f_2s
  if (pde.do_collision_operator())
  {
    // Update coeffs
    disc.compute_coefficients();

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

  auto &transformer = manager.transformer;
  auto &kronops     = manager.kronops;
  auto &moments     = manager.moments;

  auto const method = pde.options().step_method.value();

  if (manager.high_verbosity())
    node_out() << "--- begin time loop w/ dt " << pde.get_dt() << " ---\n";

  while (manager.time_step_ < manager.final_time_step_)
  {
    // take a time advance step
    auto const time           = manager.time();
    const char *time_str      = "time_advance";
    const std::string time_id = tools::timer.start(time_str);

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
          return time_advance::imex_advance<P>(manager, pde, moments, kronops, grid, transformer,
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
            return time_advance::imex_advance<P>(manager, pde, moments, kronops, grid, transformer,
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
