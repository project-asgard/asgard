#include "time_advance.hpp"
#include "adapt.hpp"
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "fast_math.hpp"
#include "solver.hpp"
#include "tools.hpp"
#ifdef ASGARD_USE_SCALAPACK
#include "cblacs_grid.hpp"
#include "scalapack_vector_info.hpp"
#endif
#include <climits>

namespace asgard::time_advance
{
template<typename P>
static fk::vector<P>
get_sources(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
            basis::wavelet_transform<P, resource::host> const &transformer,
            P const time)
{
  auto const my_subgrid = grid.get_subgrid(get_rank());
  // FIXME assume uniform degree
  auto const degree = pde.get_dimensions()[0].get_degree();
  auto const dof    = std::pow(degree, pde.num_dims) * my_subgrid.nrows();
  fk::vector<P> sources(dof);
  for (auto const &source : pde.sources)
  {
    auto const source_vect = transform_and_combine_dimensions(
        pde, source.source_funcs, grid.get_table(), transformer,
        my_subgrid.row_start, my_subgrid.row_stop, degree, time,
        source.time_func(time));
    fm::axpy(source_vect, sources);
  }
  return sources;
}

// FIXME want to change how sources/bcs are handled
template<typename P>
fk::vector<P>
adaptive_advance(method const step_method, PDE<P> &pde,
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
      return explicit_advance(pde, adaptive_grid, transformer, program_opts,
                              unscaled_parts, x_orig, time);
    case (method::imp):
      return implicit_advance(pde, adaptive_grid, transformer, program_opts,
                              unscaled_parts, x_orig, time, update_system);
    case (method::imex):
      return imex_advance(pde, adaptive_grid, transformer, program_opts,
                          unscaled_parts, x_orig, time, program_opts.solver,
                          update_system);
    };
  }

  // coarsen
  auto old_size = adaptive_grid.size();
  auto y        = adaptive_grid.coarsen_solution(pde, x_orig, program_opts);
  node_out() << " adapt -- coarsened grid from " << old_size << " -> "
             << adaptive_grid.size() << " elems\n";

  // refine
  auto refining = true;
  while (refining)
  {
    // update boundary conditions
    auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
    auto const unscaled_parts = boundary_conditions::make_unscaled_bc_parts(
        pde, adaptive_grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);

    // take a probing refinement step
    auto const y_stepped = [&]() {
      switch (step_method)
      {
      case (method::exp):
        return explicit_advance(pde, adaptive_grid, transformer, program_opts,
                                unscaled_parts, y, time);
      case (method::imp):
        return implicit_advance(pde, adaptive_grid, transformer, program_opts,
                                unscaled_parts, y, time, refining);
      case (method::imex):
        return imex_advance(pde, adaptive_grid, transformer, program_opts,
                            unscaled_parts, y, time, program_opts.solver,
                            refining);
      default:
        return fk::vector<P>();
      };
    }();

    auto const old_plan = adaptive_grid.get_distrib_plan();
    old_size            = adaptive_grid.size();
    auto const y_refined =
        adaptive_grid.refine_solution(pde, y_stepped, program_opts);
    refining = static_cast<bool>(
        get_global_max(static_cast<float>(y_stepped.size() != y_refined.size()),
                       adaptive_grid.get_distrib_plan()));

    node_out() << " adapt -- refined grid from " << old_size << " -> "
               << adaptive_grid.size() << " elems\n";

    if (!refining)
    {
      y.resize(y_stepped.size()) = y_stepped;
    }
    else
    {
      auto const y1 =
          adaptive_grid.redistribute_solution(y, old_plan, old_size);
      y.resize(y1.size()) = y1;
    }
  }

  return y;
}

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_advance(PDE<P> const &pde,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                     &unscaled_parts,
                 fk::vector<P> const &x_orig, P const time)
{
  auto const &table    = adaptive_grid.get_table();
  auto const &plan     = adaptive_grid.get_distrib_plan();
  auto const &grid     = adaptive_grid.get_subgrid(get_rank());
  auto const elem_size = element_segment_size(pde);
  auto const dt        = pde.get_dt();
  auto const col_size  = elem_size * static_cast<int64_t>(grid.ncols());
  expect(x_orig.size() == col_size);
  auto const row_size = elem_size * static_cast<int64_t>(grid.nrows());
  expect(col_size < INT_MAX);
  expect(row_size < INT_MAX);

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
  auto const apply_id = tools::timer.start("kronmult_setup");
  auto fx             = kronmult::execute(pde, table, program_opts, grid, x);

  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
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

  // -- RK step 2
  tools::timer.start(apply_id);
  fx = kronmult::execute(pde, table, program_opts, grid, x);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + c2 * dt);
    fm::axpy(sources, reduced_fx);
  }

  fk::vector<P> const bc1 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c2 * dt);
  fm::axpy(bc1, reduced_fx);

  fk::vector<P> rk_2(x_orig.size());
  exchange_results(reduced_fx, rk_2, elem_size, plan, get_rank());

  fm::copy(x_orig, x);
  P const rk_scale_2a = a31 * dt;
  P const rk_scale_2b = a32 * dt;

  fm::axpy(rk_1, x, rk_scale_2a);
  fm::axpy(rk_2, x, rk_scale_2b);

  // -- RK step 3
  tools::timer.start(apply_id);
  fx = kronmult::execute(pde, table, program_opts, grid, x);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + c3 * dt);
    fm::axpy(sources, reduced_fx);
  }

  auto const bc2 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c3 * dt);
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
implicit_advance(PDE<P> const &pde,
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
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));

#ifdef ASGARD_USE_SCALAPACK
  auto const size = elem_size * adaptive_grid.get_subgrid(get_rank()).nrows();
  fk::vector<P> x = col_to_row_major(x_orig, size);
#else
  fk::vector<P> x(x_orig);
#endif
  if (pde.num_sources > 0)
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

  if (solver != solve_opts::gmres && (first_time || update_system))
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
      return row_to_col_major(x, size_r);
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
    return row_to_col_major(x, size_r);
    ;
#else
    printf("Invalid getrs solver library specified\n");
    exit(1);
#endif
  }
  else if (solver == solve_opts::gmres)
  {
    P const tolerance  = program_opts.gmres_tolerance;
    int const restart  = program_opts.gmres_inner_iterations;
    int const max_iter = program_opts.gmres_outer_iterations;
    fk::vector<P> fx(x.size());
    solver::simple_gmres(pde, table, program_opts, grid, fx, x, fk::matrix<P>(),
                         restart, max_iter, tolerance);
    return fx;
  }
  return x;
}

// this function executes an implicit-explicit (imex) time step using the
// current solution vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
imex_advance(PDE<P> &pde, adapt::distributed_grid<P> const &adaptive_grid,
             basis::wavelet_transform<P, resource::host> const &transformer,
             options const &program_opts,
             std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                 &unscaled_parts,
             fk::vector<P> const &x_orig, P const time, solve_opts const solver,
             bool const update_system)
{
  ignore(unscaled_parts);
  ignore(solver);

  // BEFE = 0 case
  expect(time >= 0);
  expect(pde.moments.size() > 0);

  static bool first_time = true;

  // create 1D version of PDE and element table for wavelet->realspace mappings
  static PDE pde_1d = PDE(pde, PDE<P>::extract_dim0);
  static adapt::distributed_grid adaptive_grid_1d(pde_1d, program_opts);

  // Create workspace for wavelet transform
  static auto const dense_size = dense_space_size(pde_1d);
  static fk::vector<P, mem_type::owner, resource::host> workspace(dense_size *
                                                                  2);
  static std::array<fk::vector<P, mem_type::view, resource::host>, 2>
      tmp_workspace = {fk::vector<P, mem_type::view, resource::host>(
                           workspace, 0, dense_size - 1),
                       fk::vector<P, mem_type::view, resource::host>(
                           workspace, dense_size, dense_size * 2 - 1)};

  auto const &table   = adaptive_grid.get_table();
  auto const &plan    = adaptive_grid.get_distrib_plan();
  auto const dt       = pde.get_dt();
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const level     = pde.get_dimensions()[0].get_level();
  P const min         = pde.get_dimensions()[0].domain_min;
  P const max         = pde.get_dimensions()[0].domain_max;
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));

  fk::vector<P> x(x_orig);
  static auto nodes = gen_realspace_nodes(degree, level, min, max);

  auto const &grid       = adaptive_grid.get_subgrid(get_rank());
  int const A_local_rows = elem_size * grid.nrows();

  fk::vector<P> reduced_fx(A_local_rows);

  // Create moment matrices that take DG function in (x,v) and transfer to DG
  // function in x
  if (first_time || update_system)
  {
    for (auto &m : pde.moments)
    {
      m.createMomentReducedMatrix(pde, adaptive_grid.get_table());
      expect(m.get_moment_matrix().nrows() > 0);
    }
  }

  // Explicit step (f_2s)
  auto const apply_id = tools::timer.start("kronmult_setup");
  auto fx             = kronmult::execute(pde, table, program_opts, grid, x,
                              imex_flag::imex_explicit);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  fk::vector<P> f_2s(x_orig.size());
  exchange_results(reduced_fx, f_2s, elem_size, plan, get_rank());
  fm::axpy(f_2s, x, dt);

  // Create rho_2s
  fk::vector<P> mom0(dense_size);
  fk::vector<P> mom0_real(dense_size);
  fm::gemv(pde.moments[0].get_moment_matrix(), x, mom0);
  wavelet_to_realspace<P>(pde_1d, mom0, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom0_real);
  param_manager.get_parameter("n")->value = [mom0_real](P const x_v,
                                                        P const t = 0) -> P {
    ignore(t);
    return interp1(nodes, mom0_real, {x_v})[0];
  };

  // TODO: refactor into more generic function
  fk::vector<P> mom1(dense_size);
  fk::vector<P> mom1_real(dense_size);
  fm::gemv(pde.moments[1].get_moment_matrix(), x, mom1);
  wavelet_to_realspace<P>(pde_1d, mom1, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom1_real);
  param_manager.get_parameter("u")->value = [mom1_real](P const x_v,
                                                        P const t = 0) -> P {
    return interp1(nodes, mom1_real, {x_v})[0] /
           param_manager.get_parameter("n")->value(x_v, t);
  };

  fk::vector<P> mom2(dense_size);
  fk::vector<P> mom2_real(dense_size);
  fm::gemv(pde.moments[2].get_moment_matrix(), x, mom2);
  wavelet_to_realspace<P>(pde_1d, mom2, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom2_real);
  param_manager.get_parameter("theta")->value =
      [mom2_real](P const x_v, P const t = 0) -> P {
    P const u = param_manager.get_parameter("u")->value(x_v, t);
    return (interp1(nodes, mom2_real, {x_v})[0] /
            param_manager.get_parameter("n")->value(x_v, t)) -
           std::pow(u, 2);
  };

  // Update coeffs
  generate_all_coefficients<P>(pde, transformer);

  // f2 now
  P const tolerance  = program_opts.gmres_tolerance;
  int const restart  = program_opts.gmres_inner_iterations;
  int const max_iter = program_opts.gmres_outer_iterations;
  fk::vector<P> f_2(x.size());
  solver::simple_gmres(pde, table, program_opts, grid, f_2, x, fk::matrix<P>(),
                       restart, max_iter, tolerance);

  // --------------------------------
  // Third Stage
  // --------------------------------
  fm::copy(x_orig, x); // f0

  fk::vector<P> f2_x(f_2);
  fm::axpy(x, f2_x); // f0 + f2

  tools::timer.start("kronmult_setup");
  fx = kronmult::execute(pde, table, program_opts, grid, f2_x);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  fk::vector<P> f_3s(x_orig.size());
  exchange_results(reduced_fx, f_3s, elem_size, plan, get_rank());
  fm::axpy(f_3s, x, static_cast<P>(0.5) * dt); // f0 + 0.5*dt*apply_A(f0+f_2)

  tools::timer.start("kronmult_setup");
  fx = kronmult::execute(pde, table, program_opts, grid, f_2,
                         imex_flag::imex_implicit);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  exchange_results(reduced_fx, f_3s, elem_size, plan, get_rank());
  fm::axpy(f_3s, x, static_cast<P>(0.5) * dt);

  // Create rho_3s
  // TODO: refactor into more generic function
  fm::gemv(pde.moments[0].get_moment_matrix(), x, mom0);
  wavelet_to_realspace<P>(pde_1d, mom0, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom0_real);
  param_manager.get_parameter("n")->value = [mom0_real](P const x_v,
                                                        P const t = 0) -> P {
    ignore(t);
    return interp1(nodes, mom0_real, {x_v})[0];
  };

  fm::gemv(pde.moments[1].get_moment_matrix(), x, mom1);
  wavelet_to_realspace<P>(pde_1d, mom1, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom1_real);
  param_manager.get_parameter("u")->value = [mom1_real](P const x_v,
                                                        P const t = 0) -> P {
    return interp1(nodes, mom1_real, {x_v})[0] /
           param_manager.get_parameter("n")->value(x_v, t);
  };

  fm::gemv(pde.moments[2].get_moment_matrix(), x, mom2);
  wavelet_to_realspace<P>(pde_1d, mom2, adaptive_grid_1d.get_table(),
                          transformer, tmp_workspace, mom2_real);
  param_manager.get_parameter("theta")->value =
      [mom2_real](P const x_v, P const t = 0) -> P {
    P const u = param_manager.get_parameter("u")->value(x_v, t);
    return (interp1(nodes, mom2_real, {x_v})[0] /
            param_manager.get_parameter("n")->value(x_v, t)) -
           std::pow(u, 2);
  };

  // Update coeffs
  generate_all_coefficients<P>(pde, transformer);

  // Final stage f3
  fk::vector<P> f_3(x.size());
  solver::simple_gmres(pde, table, program_opts, grid, f_3, x, fk::matrix<P>(),
                       restart, max_iter, tolerance);

  return f_3;
}

template fk::vector<double> adaptive_advance(
    method const step_method, PDE<double> &pde,
    adapt::distributed_grid<double> &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts, fk::vector<double> const &x, double const time,
    bool const update_system);

template fk::vector<float> adaptive_advance(
    method const step_method, PDE<float> &pde,
    adapt::distributed_grid<float> &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts, fk::vector<float> const &x, float const time,
    bool const update_system);

template fk::vector<double> explicit_advance(
    PDE<double> const &pde,
    adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &x, double const time);

template fk::vector<float> explicit_advance(
    PDE<float> const &pde, adapt::distributed_grid<float> const &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
        &unscaled_parts,
    fk::vector<float> const &x, float const time);

template fk::vector<double> implicit_advance(
    PDE<double> const &pde,
    adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &host_space, double const time,
    bool const update_system);

template fk::vector<float> implicit_advance(
    PDE<float> const &pde, adapt::distributed_grid<float> const &adaptive_grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
        &unscaled_parts,
    fk::vector<float> const &x, float const time, bool const update_system);

template fk::vector<double> imex_advance(
    PDE<double> &pde, adapt::distributed_grid<double> const &adaptive_grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    options const &program_opts,
    std::array<boundary_conditions::unscaled_bc_parts<double>, 2> const
        &unscaled_parts,
    fk::vector<double> const &x_orig, double const time,
    solve_opts const solver, bool const update_system);

template fk::vector<float>
imex_advance(PDE<float> &pde,
             adapt::distributed_grid<float> const &adaptive_grid,
             basis::wavelet_transform<float, resource::host> const &transformer,
             options const &program_opts,
             std::array<boundary_conditions::unscaled_bc_parts<float>, 2> const
                 &unscaled_parts,
             fk::vector<float> const &x_orig, float const time,
             solve_opts const solver, bool const update_system);

} // namespace asgard::time_advance
