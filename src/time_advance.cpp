#include "time_advance.hpp"
#include "adapt.hpp"
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "fast_math.hpp"
#include "solver.hpp"
#ifdef ASGARD_USE_SCALAPACK
#include "cblacs_grid.hpp"
#include "scalapack_vector_info.hpp"
#endif
#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

namespace asgard::time_advance
{
template<typename P>
static fk::vector<P>
get_sources(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
            basis::wavelet_transform<P, resource::host> const &transformer,
            P const time)
{
  auto const my_subgrid = grid.get_subgrid(get_rank());

  auto const degree = pde.get_dimensions()[0].get_degree();
  auto const dof    = fm::ipow(degree + 1, pde.num_dims()) * my_subgrid.nrows();
  fk::vector<P> sources(dof);
  for (auto const &source : pde.sources())
  {
    auto const source_vect = transform_and_combine_dimensions(
        pde, source.source_funcs(), grid.get_table(), transformer,
        my_subgrid.row_start, my_subgrid.row_stop, degree, time,
        source.time_func()(time));
    fm::axpy(source_vect, sources);
  }
  return sources;
}

// FIXME want to change how sources/bcs are handled
template<typename P>
fk::vector<P>
adaptive_advance(method const step_method, PDE<P> &pde,
                 kron_operators<P> &operator_matrices,
                 adapt::distributed_grid<P> &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts, fk::vector<P> const &x_orig,
                 P const time, bool const update_system)
{
  if (!program_opts.do_adapt_levels)
  {
    auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
    auto const unscaled_parts = boundary_conditions::make_unscaled_bc_parts(
        pde, adaptive_grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);
    switch (step_method)
    {
    case (method::exp):
      return explicit_advance(pde, operator_matrices, adaptive_grid,
                              transformer, program_opts, unscaled_parts, x_orig,
                              time);
    case (method::imp):
      return implicit_advance(pde, operator_matrices, adaptive_grid,
                              transformer, program_opts, unscaled_parts, x_orig,
                              time, update_system);
    case (method::imex):
      return imex_advance(pde, operator_matrices, adaptive_grid, transformer,
                          program_opts, unscaled_parts, x_orig, fk::vector<P>(),
                          time, program_opts.solver, update_system);
    };
  }

  // coarsen
  auto old_size = adaptive_grid.size();
  auto y        = adaptive_grid.coarsen_solution(pde, x_orig, program_opts);
  node_out() << " adapt -- coarsened grid from " << old_size << " -> "
             << adaptive_grid.size() << " elems\n";

  // clear the matrices if the coarsening removed indexes
  if (old_size != adaptive_grid.size())
    operator_matrices.clear();

  // save coarsen stats
  pde.adapt_info.initial_dof = old_size;
  pde.adapt_info.coarsen_dof = adaptive_grid.size();
  pde.adapt_info.refine_dofs = std::vector<int>();
  // save GMRES stats starting with the coarsen stats
  pde.adapt_info.gmres_stats =
      std::vector<std::vector<gmres_info<P>>>({pde.gmres_outputs});

  // refine
  bool refining = true;
  fk::vector<P> y_first_refine;
  while (refining)
  {
    // update boundary conditions
    auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
    auto const unscaled_parts = boundary_conditions::make_unscaled_bc_parts(
        pde, adaptive_grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);

    // take a probing refinement step
    fk::vector<P> y_stepped = [&]() {
      switch (step_method)
      {
      case (method::exp):
        return explicit_advance(pde, operator_matrices, adaptive_grid,
                                transformer, program_opts, unscaled_parts, y,
                                time);
      case (method::imp):
        return implicit_advance(pde, operator_matrices, adaptive_grid,
                                transformer, program_opts, unscaled_parts, y,
                                time, refining);
      case (method::imex):
        return imex_advance(pde, operator_matrices, adaptive_grid, transformer,
                            program_opts, unscaled_parts, y, y_first_refine,
                            time, program_opts.solver, refining);
      default:
        return fk::vector<P>();
      };
    }();

    auto const old_plan = adaptive_grid.get_distrib_plan();
    old_size            = adaptive_grid.size();
    fk::vector<P> y_refined =
        adaptive_grid.refine_solution(pde, y_stepped, program_opts);
    // if either one of the ranks reports 1, i.e., y_stepped.size() changed
    refining = get_global_max<bool>(y_stepped.size() != y_refined.size(),
                                    adaptive_grid.get_distrib_plan());

    node_out() << " adapt -- refined grid from " << old_size << " -> "
               << adaptive_grid.size() << " elems\n";
    // save refined DOF stats
    pde.adapt_info.refine_dofs.push_back(adaptive_grid.size());
    // append GMRES stats for refinement
    pde.adapt_info.gmres_stats.push_back({pde.gmres_outputs});

    if (!refining)
    {
      y = std::move(y_stepped);
    }
    else
    {
      // added more indexes, matrices will have to be remade
      operator_matrices.clear();

      y = adaptive_grid.redistribute_solution(y, old_plan, old_size);

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
}

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_advance(PDE<P> const &pde, kron_operators<P> &operator_matrices,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                     &unscaled_parts,
                 fk::vector<P> const &x_orig, P const time)
{
  auto const &plan     = adaptive_grid.get_distrib_plan();
  auto const &grid     = adaptive_grid.get_subgrid(get_rank());
  auto const elem_size = element_segment_size(pde);
  auto const dt        = pde.get_dt();
  auto const col_size  = elem_size * static_cast<int64_t>(grid.ncols());
  expect(x_orig.size() == col_size);
  auto const row_size = elem_size * static_cast<int64_t>(grid.nrows());
  expect(col_size < INT_MAX);
  expect(row_size < INT_MAX);

  operator_matrices.make(imex_flag::unspecified, pde, adaptive_grid, program_opts);

  // time advance working vectors
  // input vector for apply_A
  fk::vector<P> x(x_orig);
  // a buffer for reducing across subgrid row
  fk::vector<P> reduced_fx(row_size);

  expect(time >= 0);
  expect(dt > 0);

  // see
  // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
  P const a21 = 0.5;
  P const a31 = -1.0;
  P const a32 = 2.0;
  P const b1  = 1.0 / 6.0;
  P const b2  = 2.0 / 3.0;
  P const b3  = 1.0 / 6.0;
  P const c2  = 1.0 / 2.0;
  P const c3  = 1.0;

  // FIXME eventually want to extract RK step into function
  // -- RK step 1
  fk::vector<P> fx(row_size);
  {
    tools::time_event performance(
        "kronmult", operator_matrices.flops(imex_flag::unspecified));
    operator_matrices.apply(imex_flag::unspecified, time, 1.0, x.data(), 0.0, fx.data());
  }
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources() > 0)
  {
    auto const sources = get_sources(pde, adaptive_grid, transformer, time);
    fm::axpy(sources, reduced_fx);
  }

  auto const bc0 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time);
  fm::axpy(bc0, reduced_fx);

  // FIXME I eventually want to return a vect here
  fk::vector<P> rk_1(x_orig.size());
  exchange_results(reduced_fx, rk_1, elem_size, plan, get_rank());
  P const rk_scale_1 = a21 * dt;
  fm::axpy(rk_1, x, rk_scale_1);

  P const t2 = time + c2 * dt;
  {
    tools::time_event performance(
        "kronmult", operator_matrices.flops(imex_flag::unspecified));
    operator_matrices.apply(imex_flag::unspecified, t2, 1.0, x.data(), 0.0, fx.data());
  }
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources() > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, t2);
    fm::axpy(sources, reduced_fx);
  }

  fk::vector<P> const bc1 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      t2);
  fm::axpy(bc1, reduced_fx);

  fk::vector<P> rk_2(x_orig.size());
  exchange_results(reduced_fx, rk_2, elem_size, plan, get_rank());

  fm::copy(x_orig, x);
  P const rk_scale_2a = a31 * dt;
  P const rk_scale_2b = a32 * dt;

  fm::axpy(rk_1, x, rk_scale_2a);
  fm::axpy(rk_2, x, rk_scale_2b);

  // -- RK step 3
  P const t3 = time + c3 * dt;
  {
    tools::time_event performance(
        "kronmult", operator_matrices.flops(imex_flag::unspecified));
    operator_matrices.apply(imex_flag::unspecified, t3, 1.0, x.data(), 0.0, fx.data());
  }
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources() > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, t3);
    fm::axpy(sources, reduced_fx);
  }

  auto const bc2 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      t3);
  fm::axpy(bc2, reduced_fx);

  fk::vector<P> rk_3(x_orig.size());
  exchange_results(reduced_fx, rk_3, elem_size, plan, get_rank());

  // -- finish
  fm::copy(x_orig, x);
  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::axpy(rk_1, x, scale_1);
  fm::axpy(rk_2, x, scale_2);
  fm::axpy(rk_3, x, scale_3);

  return x;
}

// this function executes an implicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
implicit_advance(PDE<P> &pde, kron_operators<P> &operator_matrices,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                     &unscaled_parts,
                 fk::vector<P> const &x_orig, P const time,
                 bool const update_system)
{
  expect(time >= 0);
#ifdef ASGARD_USE_SCALAPACK
  auto sgrid = get_grid();
#endif
  solve_opts const solver = program_opts.solver;
  static fk::matrix<P, mem_type::owner, resource::host> A;
  static std::vector<int> ipiv;
  static bool first_time = true;

  auto const &table   = adaptive_grid.get_table();
  auto const dt       = pde.get_dt();
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = fm::ipow(degree + 1, pde.num_dims());

#ifdef ASGARD_USE_SCALAPACK
  auto const size = elem_size * adaptive_grid.get_subgrid(get_rank()).nrows();
  fk::vector<P> x(size);
  exchange_results(x_orig, x, size, adaptive_grid.get_distrib_plan(), get_rank());
#else
  fk::vector<P> x(x_orig);
#endif
  if (pde.num_sources() > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + dt);
    fm::axpy(sources, x, dt);
  }

  auto const &grid       = adaptive_grid.get_subgrid(get_rank());
  int const A_local_rows = elem_size * grid.nrows();
  int const A_local_cols = elem_size * grid.ncols();
#ifdef ASGARD_USE_SCALAPACK
  int nm = A_local_rows;
  bcast(&nm, 1, 0);
  int const A_global_size = elem_size * table.size();
  assert(x.size() <= nm);

  fk::scalapack_vector_info vinfo(A_global_size, nm, sgrid);
  fk::scalapack_matrix_info minfo(A_global_size, A_global_size, nm, nm, sgrid);
#endif
  auto const bc = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + dt);
  fm::axpy(bc, x, dt);

  if ((solver != solve_opts::gmres && solver != solve_opts::bicgstab) && (first_time || update_system))
  {
    first_time = false;

    A.clear_and_resize(A_local_rows, A_local_cols);
    build_system_matrix(pde, table, A, grid);

    // AA = I - dt*A;
    fm::scal(-dt, A);
    if (grid.row_start == grid.col_start)
    {
      for (int i = 0; i < A.nrows(); ++i)
      {
        A(i, i) += 1.0;
      }
    }
    int ipiv_size{0};
    if (solver == solve_opts::direct)
    {
      ipiv_size = A.nrows();
      if (ipiv.size() != static_cast<unsigned long>(ipiv_size))
      {
        ipiv.resize(ipiv_size);
      }
      fm::gesv(A, x, ipiv);
      return x;
    }
    else if (solver == solve_opts::scalapack)
    {
#ifdef ASGARD_USE_SCALAPACK
      ipiv_size = minfo.local_rows() + minfo.mb();
      if (ipiv.size() != static_cast<unsigned long>(ipiv_size))
      {
        ipiv.resize(ipiv_size);
      }
      fm::gesv(A, minfo, x, vinfo, ipiv);
      auto const size_r =
          elem_size * adaptive_grid.get_subgrid(get_rank()).ncols();
      fk::vector<P> tmp(size_r);
      exchange_results(x, tmp, size_r, adaptive_grid.get_distrib_plan(), get_rank());
      return tmp;
#else
      printf("Invalid gesv solver library specified\n");
      exit(1);
      return x;
#endif
    }
  } // end first time/update system

  if (solver == solve_opts::direct)
  {
    fm::getrs(A, x, ipiv);
    return x;
  }
  else if (solver == solve_opts::scalapack)
  {
#ifdef ASGARD_USE_SCALAPACK
    fm::getrs(A, minfo, x, vinfo, ipiv);
    auto const size_r =
        elem_size * adaptive_grid.get_subgrid(get_rank()).ncols();
    fk::vector<P> tmp(size_r);
    exchange_results(x, tmp, size_r, adaptive_grid.get_distrib_plan(), get_rank());
    return tmp;
#else
    printf("Invalid getrs solver library specified\n");
    exit(1);
#endif
  }
  else if (solver == solve_opts::gmres)
  {
    operator_matrices.make(imex_flag::unspecified, pde, adaptive_grid,
                           program_opts);
    P const tolerance  = program_opts.gmres_tolerance;
    int const restart  = program_opts.gmres_inner_iterations;
    int const max_iter = program_opts.gmres_outer_iterations;
    fk::vector<P> fx(x);
    // TODO: do something better to save gmres output to pde
    pde.gmres_outputs[0] = solver::simple_gmres_euler<P, resource::host>(
        pde.get_dt(), imex_flag::unspecified, operator_matrices,
        fx, x, restart, max_iter, tolerance);
    return fx;
  }
  else if (solver == solve_opts::bicgstab)
  {
    operator_matrices.make(imex_flag::unspecified, pde, adaptive_grid,
                           program_opts);
    P const tolerance  = program_opts.gmres_tolerance;
    int const max_iter = program_opts.gmres_outer_iterations;
    fk::vector<P> fx(x);
    // TODO: do something better to save gmres output to pde
    pde.gmres_outputs[0] = solver::bicgstab_euler<P, resource::host>(
        pde.get_dt(), imex_flag::unspecified, operator_matrices,
        fx, x, max_iter, tolerance);
    return fx;
  }
  return x;
}

asgard::parser make_parser(std::vector<std::string> const arguments)
{
  std::vector<char *> argv;
  argv.push_back(const_cast<char *>("asgard"));
  for (const auto &arg : arguments)
  {
    argv.push_back(const_cast<char *>(arg.data()));
  }
  argv.push_back(nullptr);

  return asgard::parser(argv.size() - 1, argv.data());
}

asgard::options make_options(std::vector<std::string> const arguments)
{
  return asgard::options(make_parser(arguments));
}

// this function executes an implicit-explicit (imex) time step using the
// current solution vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
imex_advance(PDE<P> &pde, kron_operators<P> &operator_matrices,
             adapt::distributed_grid<P> const &adaptive_grid,
             basis::wavelet_transform<P, resource::host> const &transformer,
             options const &program_opts,
             std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                 &unscaled_parts,
             fk::vector<P> const &f_0, fk::vector<P> const &x_prev,
             P const time, solve_opts const solver, bool const update_system)
{
  ignore(unscaled_parts);
  ignore(solver);

  // BEFE = 0 case
  expect(time >= 0);
  expect(pde.moments.size() > 0);

  static bool first_time = true;

  // create 1D version of PDE and element table for wavelet->realspace mappings
  PDE pde_1d       = PDE(pde, PDE<P>::extract_dim0);
  int const degree = pde.get_dimensions()[0].get_degree();
  int const level  = pde.get_dimensions()[0].get_level();

  options const opts_1d = make_options(
      {"-d " + std::to_string(degree), "-f", "-l " + std::to_string(level),
       "-m " + std::to_string(program_opts.max_level)});
  adapt::distributed_grid adaptive_grid_1d(pde_1d, opts_1d);

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
  int const N_elements = fm::two_raised_to(level);

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

  // Create moment matrices that take DG function in (x,v) and transfer to DG
  // function in x
  if (first_time || update_system)
  {
    tools::time_event performance("update_system");
    for (auto &m : pde.moments)
    {
      m.createFlist(pde, program_opts);
      expect(m.get_fList().size() > 0);

      m.createMomentVector(pde, program_opts, adaptive_grid.get_table());
      expect(m.get_vector().size() > 0);

      m.createMomentReducedMatrix(pde, adaptive_grid.get_table());
      // expect(m.get_moment_matrix().nrows() > 0);
    }

    if (pde.do_poisson_solve())
    {
      // Setup poisson matrix initially
      solver::setup_poisson(N_elements, min, max, pde.poisson_diag,
                            pde.poisson_off_diag);
    }

    pde.E_field.resize(quad_dense_size);
    pde.phi.resize(quad_dense_size);
    pde.E_source.resize(quad_dense_size);

    first_time = false;
  }

  auto do_poisson_update = [&](fk::vector<P, mem_type::owner, imex_resrc> const
                                   &f_in) {
    fk::vector<P> poisson_source(quad_dense_size);
    fk::vector<P> phi(quad_dense_size);
    fk::vector<P> poisson_E(quad_dense_size);
    {
      tools::time_event pupdate_("poisson_update");
      // Get 0th moment
      fk::vector<P, mem_type::owner, imex_resrc> mom0(dense_size);
      fm::sparse_gemv(pde.moments[0].get_moment_matrix_dev(), f_in, mom0);
      fk::vector<P> &mom0_real = pde.moments[0].create_realspace_moment(
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

    // Update coeffs
    generate_all_coefficients<P>(pde, transformer);

#ifdef ASGARD_USE_MATLAB
    auto &ml_plot = ml::matlab_plot::get_instance();
    // TODO: add plot_freq check if keeping this much longer
    ml_plot.reset_params();
    ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, nodes);
    ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, poisson_E);
    ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, poisson_source);
    ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, phi);
    ml_plot.add_param(time);
    ml_plot.call("electric");
#endif
  };

  auto calculate_moments =
      [&](fk::vector<P, mem_type::owner, imex_resrc> const &f_in) {
        // \int f dv
        fk::vector<P, mem_type::owner, imex_resrc> mom0(dense_size);
        fm::sparse_gemv(pde.moments[0].get_moment_matrix_dev(), f_in, mom0);
        fk::vector<P> &mom0_real = pde.moments[0].create_realspace_moment(
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
        fm::sparse_gemv(pde.moments[1].get_moment_matrix_dev(), f_in, mom1);
        fk::vector<P> &mom1_real = pde.moments[1].create_realspace_moment(
            pde_1d, mom1, adaptive_grid_1d.get_table(), transformer,
            tmp_workspace);

        // u_x = \int f v_x  dv / n
        param_manager.get_parameter("u")->value = [&](P const x_v,
                                                      P const t = 0) -> P {
          return interp1(nodes, mom1_real, {x_v})[0] /
                 param_manager.get_parameter("n")->value(x_v, t);
        };
        if (pde.num_dims() == 3 && pde.moments.size() > 3)
        {
          // Calculate additional moments for PDEs with more than one velocity
          // dimension

          // \int f v_{y} dv
          fk::vector<P, mem_type::owner, imex_resrc> mom2(dense_size);
          fm::sparse_gemv(pde.moments[2].get_moment_matrix_dev(), f_in, mom2);
          fk::vector<P> &mom2_real = pde.moments[2].create_realspace_moment(
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
          fm::sparse_gemv(pde.moments[3].get_moment_matrix_dev(), f_in, mom3);
          fk::vector<P> &mom3_real = pde.moments[3].create_realspace_moment(
              pde_1d, mom3, adaptive_grid_1d.get_table(), transformer,
              tmp_workspace);

          // \int f v_y^2 dv
          fk::vector<P, mem_type::owner, imex_resrc> mom4(dense_size);
          fm::sparse_gemv(pde.moments[4].get_moment_matrix_dev(), f_in, mom4);
          fk::vector<P> &mom4_real = pde.moments[4].create_realspace_moment(
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
        else if (pde.num_dims() == 4 && pde.moments.size() > 6)
        {
          // Moments for 1X3V case
          // TODO: this will be refactored to replace dimension cases in the
          // future
          std::vector<fk::vector<P, mem_type::owner, imex_resrc>> moments;
          std::vector<fk::vector<P> *> moments_real;
          // Create moment matrices and realspace moments for all moments in PDE
          for (size_t mom = 2; mom < pde.moments.size(); mom++)
          {
            // \int f v_x dv
            moments.push_back(
                fk::vector<P, mem_type::owner, imex_resrc>(dense_size));
            fm::sparse_gemv(pde.moments[mom].get_moment_matrix_dev(), f_in,
                            moments.back());
            moments_real.push_back(&pde.moments[mom].create_realspace_moment(
                pde_1d, moments.back(), adaptive_grid_1d.get_table(),
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
          fm::sparse_gemv(pde.moments[2].get_moment_matrix_dev(), f_in, mom2);
          fk::vector<P> &mom2_real = pde.moments[2].create_realspace_moment(
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
                                       adaptive_grid, program_opts);

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
  P const tolerance  = program_opts.gmres_tolerance;
  int const restart  = program_opts.gmres_inner_iterations;
  int const max_iter = program_opts.gmres_outer_iterations;
  fk::vector<P, mem_type::owner, imex_resrc> f_1(f.size());
  fk::vector<P, mem_type::owner, imex_resrc> f_1_output(f.size());
  if (pde.do_collision_operator())
  {
    // Update coeffs
    generate_all_coefficients<P>(pde, transformer);

    // f2 now
    operator_matrices.reset_coefficients(imex_flag::imex_implicit, pde,
                                         adaptive_grid, program_opts);

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
                                       adaptive_grid, program_opts);

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
    tools::timer.start("implicit_2_coeff");
    generate_all_coefficients<P>(pde, transformer);
    tools::timer.stop("implicit_2_coeff");

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
                                         adaptive_grid, program_opts);

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

#ifdef ASGARD_ENABLE_DOUBLE
template fk::vector<double> adaptive_advance(
    method const step_method, PDE<double> &pde,
    kron_operators<double> &operator_matrix,
    adapt::distributed_grid<double> &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts, fk::vector<double> const &x, double const time,
    bool const update_system);

template fk::vector<double> explicit_advance(
    PDE<double> const &pde, kron_operators<double> &operator_matrix,
    adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &x, double const time);

template fk::vector<double> implicit_advance(
    PDE<double> &pde, kron_operators<double> &operator_matrix,
    adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &host_space, double const time,
    bool const update_system);

template fk::vector<double> imex_advance(
    PDE<double> &pde, kron_operators<double> &operator_matrix,
    adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &f_0, fk::vector<double> const &x_prev,
    double const time, solve_opts const solver, bool const update_system);

#endif

#ifdef ASGARD_ENABLE_FLOAT

template fk::vector<float> adaptive_advance(
    method const step_method, PDE<float> &pde,
    kron_operators<float> &operator_matrix,
    adapt::distributed_grid<float> &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts, fk::vector<float> const &x, float const time,
    bool const update_system);

template fk::vector<float> explicit_advance(
    PDE<float> const &pde, kron_operators<float> &operator_matrix,
    adapt::distributed_grid<float> const &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
        &unscaled_parts,
    fk::vector<float> const &x, float const time);

template fk::vector<float> implicit_advance(
    PDE<float> &pde, kron_operators<float> &operator_matrix,
    adapt::distributed_grid<float> const &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
        &unscaled_parts,
    fk::vector<float> const &x, float const time, bool const update_system);

template fk::vector<float>
imex_advance(PDE<float> &pde, kron_operators<float> &operator_matrix,
             adapt::distributed_grid<float> const &adaptive_grid,
             basis::wavelet_transform<float, resource::host> const &transformer,
             options const &program_opts,
             std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
                 &unscaled_parts,
             fk::vector<float> const &f_0, fk::vector<float> const &x_prev,
             float const time, solve_opts const solver,
             bool const update_system);
#endif

} // namespace asgard::time_advance
