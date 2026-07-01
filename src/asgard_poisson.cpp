#include "asgard_poisson.hpp"
#include "asgard_solver.hpp"
#include "asgard_small_mats.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard
{

template<typename P>
poisson_md<P>::poisson_md(int const num_pos, int const max_level, std::array<P, max_num_dimensions> const &xleft,
                          std::array<P, max_num_dimensions> const &xright, connection_patterns const &conn,
                          hierarchy_manipulator<P> const &hier, moments_list const &mlist,
                          build_term_func<P> build, iter_solve_func<P> iter_solve_func, moment_id const m0)
  : num_dims(num_pos), pdof(hier.degree() + 1), mom0(m0), iter_solve(iter_solve_func)
{
  rassert((num_dims > 1) and (num_dims <= max_pos_dims), "poisson_md should only be used for 2 or 3 spatial dimensions");
  int const nelem = fm::ipow2(max_level);

  // set up scaling for each direction
  for (int const d : iindexof(num_dims))
    derivative_scale[d] = static_cast<P>(nelem) / (xright[d] - xleft[d]);

  // set up moms_electric
  for (int const d : iindexof(num_dims))
    moms_electric[d] = mlist.get_check_id(moment::electric(dimension_id(d), num_pos));

  // set up the terms for the laplician: laplacian(f) = div(grad(f))
  term_1d<P> div = term_div<P>(-1, flux_type::upwind, boundary_type::periodic);
  term_1d<P> grad = term_grad<P>(1, flux_type::upwind, boundary_type::periodic);

  // the multi-dimensional Laplacian, initially set to identity in all dimensions
  std::vector<term_1d<P>> ops(num_dims, term_identity{});
  laplacian_terms.resize(num_dims);
  for (int const d : iindexof(num_dims))
  {
    term_1d<P> laplacian({div, grad});
    laplacian.set_penalty(derivative_scale[d]);
    ops[d] = laplacian; // using operator in the d-direction
    term_md<P> laplacian_md(ops);
    laplacian_terms[d] = term_entry<P>(std::move(laplacian_md));
    ops[d] = term_identity{}; // reset back to identity
  }

  // build the term entries from the laplacian terms
  tools::time_event timing_("initial poisson_md coefficients mats");
  for (auto &tentry : laplacian_terms) {
    for (int const d : iindexof(num_dims))
      build(tentry, d, max_level);
  }

  // set up the derivative matrix
  vector2d<double> p2d = legendre::poly2diff(hier.degree());
  block_diag_matrix<P> diag(pdof * pdof, nelem);
  if constexpr (is_double<P>)
    fill_pattern(p2d[0], diag);
  else {
    std::vector<P> fp2d(pdof * pdof);
    std::copy_n(p2d[0], pdof * pdof, fp2d.begin());
    fill_pattern(fp2d.data(), diag);
  }
  derivative_mat = hier.diag2hierarchical(diag, max_level, conn);

  #ifdef ASGARD_USE_GPU
  #ifdef ASGARD_GPU_MEMGREEDY
  gpu_derivative_mat = derivative_mat.data_vector();
  #else
  compute->set_device(gpu::device{0});
  gpul_derivative_mat.resize(max_level + 1);
  std::vector<P*> coeff_pntrs(max_level + 1, nullptr);
  for (int l = 0; l < max_level; l++) {
    gpul_derivative_mat[l] = derivative_mat.get_subpattern(l, conn).data_vector();
    coeff_pntrs[l] = gpul_derivative_mat[l].data();
  }
  gpul_derivative_mat[max_level] = derivative_mat.data_vector();
  coeff_pntrs[max_level] = gpul_derivative_mat[max_level].data();
  gpu_derivative_mat = coeff_pntrs;
  #endif
  gpu_density0.resize(1);
  #endif
}

#ifndef ASGARD_USE_GPU
template<typename P>
void poisson_md<P>::solve(std::vector<P> &density, momentset<P> &moms,
                          sparse_grid const &position_grid, connection_patterns const &conn,
                          kronmult::workspace<P> &work, poisson_bc const bc)
{
  tools::time_event psolve_("poisson_md");
  
  rassert(operator bool(), "poisson_md must be initialized before solve");
  size_t const n = density.size();

  solve_potential_(density, position_grid, conn, work, bc);

  // get the electric field from the potential
  for (int const d : iindexof(num_dims))
  {
    moment_id const &mid = moms_electric[d];
    if (mid == moment_id::unset()) continue;
    moms[mid].resize(n);
    kronmult::permutes perm = std::vector<int>{d, };
    block_cpu(pdof, position_grid, conn, perm, derivative_mat,
              derivative_scale[d], potential.data(), P{0.0}, moms[mid].data(), work);
  }
}

template<typename P>
void poisson_md<P>::solve_potential_(std::vector<P> &density, sparse_grid const &position_grid,
                                     connection_patterns const &conn, kronmult::workspace<P> &work, poisson_bc const bc)
{
  size_t const n = density.size();

  // Resize efield and set an initial guess of zeros
  potential.resize(n);
  std::fill(potential.begin(), potential.end(), P{0.0});

  // Save the average density to restore it later
  P density0 = density[0];

  // For periodic boundaries, the solution is only unique up to a constant.
  // We must force the mean of the RHS to 0 so the iterative solver doesn't blow up.
  if (bc == poisson_bc::periodic) {
    density[0] = P{0};
  }

  // Define the Matrix-Vector Product (The LHS)
  auto apply_lhs = [&](P alpha, P const x[], P beta, P y[]) -> void
  {
    tools::time_event performance_("poisson_md kronmult");
    P b = beta; // on first iteration, overwrite y

    for (term_entry<P> const &tentry: laplacian_terms)
    {
      block_cpu(pdof, position_grid, conn, tentry.perm, tentry.coeffs, alpha, x, b, y, work);
      b = 1; // next iteration appends on y
    }
  };

  // Execute the Solve
  int num_iter = iter_solve(apply_lhs, density, potential);
  std::ignore = num_iter;

  // Restore the density
  if (bc == poisson_bc::periodic)
    density[0] = density0;
}
#endif

#ifdef ASGARD_USE_GPU
template<typename P>
void poisson_md<P>::solve(gpu::vector<P> &density, sparse_grid const &position_grid,
                          connection_patterns const &conn, interpolate_func<P> interpolate,
                          kronmult::workspace<P> &work, poisson_bc const bc)
{
  tools::time_event psolve_("poisson_md GPU");
  
  rassert(operator bool(), "poisson_md must be initialized before solve");
  size_t const n = position_grid.num_dof();
  size_t const np = work.gpu_w1[0].size();
  work.gpu_w1[0].resize(n);

  solve_potential_(density, position_grid, conn, work, bc);

  // get the electric field from the potential
  gpu_efield.resize(n);
  for (int const d : iindexof(num_dims))
  {
    moment_id const mid = moms_electric[d];
    if (mid == moment_id::unset()) continue;
    kronmult::permutes perm = std::vector<int>{d, };
    block_gpu(gpu::device{0}, pdof, position_grid, conn, perm, gpu_derivative_mat,
              derivative_scale[d], gpu_potential.data(), P{0.0}, gpu_efield.data(), work, derivative_mat);

    // interpolate electric field moment
    interpolate(gpu_efield, mid);
  }
  work.gpu_w1[0].resize(np);
}

template<typename P>
void poisson_md<P>::solve_potential_(gpu::vector<P> &density, sparse_grid const &position_grid,
                                     connection_patterns const &conn, kronmult::workspace<P> &work, poisson_bc const bc)
{
  size_t const n = position_grid.num_dof();

  // Resize efield and set an initial guess of zeros
  gpu_potential.resize(n);
  gpu::fill_zeros(n, gpu_potential.data());

  // For periodic boundaries, the solution is only unique up to a constant.
  // We must force the mean of the RHS to 0 so the iterative solver doesn't blow up.
  if (bc == poisson_bc::periodic) {
    gpu::memcopy_dev2dev(1, density.data(), gpu_density0.data());
    gpu::fill_zeros(1, density.data());
  }
  
  // Define the Matrix-Vector Product (The LHS)
  auto apply_lhs = [&](P alpha, P const x[], P beta, P y[]) -> void
  {
    tools::time_event performance_("poisson_md kronmult");
    P b = beta; // on first iteration, overwrite y

    for (term_entry<P> const &tentry: laplacian_terms)
    {
      block_gpu(gpu::device{0}, pdof, position_grid, conn, tentry.perm, tentry.gpu_coeffs,
                alpha, x, b, y, work, tentry.coeffs);
      b = 1; // next iteration appends on y
    }
  };

  // Execute the Solve
  compute->set_device(gpu::device{0});
  int num_iter = iter_solve(apply_lhs, density, gpu_potential);
  std::ignore = num_iter;

  // Restore the density
  if (bc == poisson_bc::periodic)
    gpu::memcopy_dev2dev(1, gpu_density0.data(), density.data());
}
#endif

template<typename P>
void poisson_1d<P>::solve(std::vector<P> const &density, P dleft, P dright,
                       poisson_bc const bc, std::vector<P> &efield)
{
  tools::time_event psolve_("poisson_1d");

  if (current_level == 0)
  {
    efield.resize(1);
    efield[0] = -(dright - dleft) / (xmax - xmin);
    return;
  }

  int const nelem = fm::ipow2(current_level);

  P const dx = (xmax - xmin) / static_cast<P>(nelem);

  int const nnodes = nelem - 1;

  // integrals of hat-basis functions x, 1-x vs Legendre basis 1, sqrt(3) * 2x-1
  // over canonical element (0, 1)
  // the input coefficients are l-2 normalized over sub-cells, hence the sqrt-scaling
  P const c0 = std::sqrt(dx) * 0.5;
  P const c1 = std::sqrt(dx) * std::sqrt(3.0) * P{1} / P{6}; // the integral with the left basis is negative

  span2d<P const> rho(pdof, nelem, density.data());

  // building the right-hand-side vector
  if (bc == poisson_bc::periodic)
  {
    dleft = dright = P{0};

    P average = 0;
    for (int i : iindexof(nelem))
      average += rho[i][0]; // reading the constant
    // the extra 2 below is because the correction is applied to 2 elements
    average *= P{2} * dx / (xmax - xmin);

    if (pdof == 1) { // consider only constant functions
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0] - average);
    } else {
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0] - average)
                + c1 * rho[i][1] - c1 * rho[i + 1][1];
    }
  }
  else
  {
    if (pdof == 1) { // consider only constant functions
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0]);
    } else {
      for (int i : iindexof(nnodes))
        rhs[i] = c0 * (rho[i][0] + rho[i + 1][0])
                + c1 * rho[i][1] - c1 * rho[i + 1][1];
    }
    rhs.front() += dleft / dx;
    rhs.back()  += dright / dx;
  }

  // Linear Solve //
  compute->pttrs(diag, subdiag, rhs);

  // Set Potential and Electric Field in DG Nodes //
  efield.resize(nelem);

  efield[0] = - (rhs[0] - dleft) / dx;
  for (int i = 1; i < nelem - 1; i++)
    efield[i] = - (rhs[i] - rhs[i - 1]) / dx;
  efield.back() = - (dright - rhs.back()) / dx;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class poisson_md<double>;
template class poisson_1d<double>;
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class poisson_md<float>;
template class poisson_1d<float>;
#endif //ASGARD_ENABLE_FLOAT

} // namespace asgard::poisson