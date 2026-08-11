#include "asgard_poisson.hpp"
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
                          build_term_func<P> build, moment_id const m0, prog_opts const &opts)
  : num_dims(num_pos), pdof(hier.degree() + 1), mom0(m0),
    cg_solver(opts.poisson_tolerance.value(), opts.poisson_iterations.value()),
    precon(opts.poisson_precon.value_or(precon_method::none))
{
  assert((num_dims > 1) and (num_dims <= max_pos_dims));
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
              -derivative_scale[d], potential.data(), P{0.0}, moms[mid].data(), work);
  }
}

template<typename P>
void poisson_md<P>::remap_(indexset const& iset_old, indexset const &iset_new, std::vector<P> &x) const
{
  int64_t const num_old = iset_old.num_indexes();
  int64_t const num_new = iset_new.num_indexes();
  int const block_size = fm::ipow(pdof, num_dims);

  // Initialize exactly to 0 to automatically handle newly refined points
  std::vector<P> x_new(num_new * block_size, P{0});

  int64_t iold = 0;
  int64_t inew = 0;

  enum class index_relation
  {
    asameb,
    abeforeb,
    bbeforea
  };

  auto compare_indexes = [&](int const a[], int const b[]) -> index_relation
    {
      for(int const d : iindexof(num_dims)) {
        if (a[d] < b[d]) return index_relation::abeforeb;
        if (a[d] > b[d]) return index_relation::bbeforea;
      }
      return index_relation::asameb;
    };

  while (inew < num_new && iold < num_old) {
    index_relation relation = compare_indexes(iset_new[inew], iset_old[iold]);

    if (relation == index_relation::asameb) {
      // Point survived adaptation: Transfer data
      std::copy_n(x.data() + iold * block_size, block_size, x_new.data() + inew * block_size);
      inew++;
      iold++;
    } else if (relation == index_relation::abeforeb) {
      // New point added (Refinement): Already 0, just advance
      inew++;
    } else {
      // Old point removed (Coarsening): Discard data
      iold++;
    }
  }
  std::swap(x, x_new);
}

template<typename P>
void poisson_md<P>::solve_potential_(std::vector<P> &density, sparse_grid const &position_grid,
                                     connection_patterns const &conn, kronmult::workspace<P> &work, poisson_bc const bc)
{
  // Remap the previous potential to the new grid if needed to use as a warm start
  if (generation != position_grid.generation()) {
    remap_(iset_, position_grid.iset(), potential);
    iset_ = position_grid.iset();
    generation = position_grid.generation();
  }

  // Save the average density to restore it later
  P density0 = density[0];

  // For periodic boundaries, the solution is only unique up to a constant.
  // We must force the mean of the RHS to 0 so the iterative solver doesn't blow up.
  if (bc == poisson_bc::periodic)
    density[0] = P{0};

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
  int num_iter = 0;
  if (precon.method() == precon_method::jacobi) {
    num_iter = cg_solver.solve([&](P y[]) -> void
      {
        tools::time_event timing_("poisson_md jacobi preconditioner");
        fm::jacobi_apply(density.size(), precon.jacobi(), y);
      }, apply_lhs, density, potential);
  } else {
    num_iter = cg_solver.solve(nullptr, apply_lhs, density, potential);
  }
  // std::cout << num_iter << " iterations\n";
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

  solve_potential_(density, position_grid, conn, work, bc);

  // get the electric field from the potential
  gpu_efield.resize(density.size());
  for (int const d : iindexof(num_dims))
  {
    moment_id const mid = moms_electric[d];
    if (mid == moment_id::unset()) continue;
    kronmult::permutes perm = std::vector<int>{d, };
    block_gpu(gpu::device{0}, pdof, position_grid, conn, perm, gpu_derivative_mat,
              -derivative_scale[d], gpu_potential.data(), P{0.0}, gpu_efield.data(), work, derivative_mat);

    // interpolate electric field moment
    interpolate(gpu_efield, mid);
  }
}

template<typename P>
void poisson_md<P>::remap_(indexset const& iset_old, indexset const &iset_new, gpu::vector<P> &x) const
{
  int64_t const num_old = iset_old.num_indexes();
  int64_t const num_new = iset_new.num_indexes();
  int const block_size = fm::ipow(pdof, num_dims);

  // Initialize exactly to 0.0 to automatically handle newly refined points
  gpu::vector<P> x_new(num_new * block_size);
  gpu::fill_zeros(x_new.size(), x_new.data());
  std::vector<int64_t> transfers;
  transfers.reserve(num_old + num_new);

  int64_t iold = 0;
  int64_t inew = 0;

  enum class index_relation
  {
    asameb,
    abeforeb,
    bbeforea
  };

  auto compare_indexes = [&](int const a[], int const b[]) -> index_relation
    {
      for(int const d : iindexof(num_dims)) {
        if (a[d] < b[d]) return index_relation::abeforeb;
        if (a[d] > b[d]) return index_relation::bbeforea;
      }
      return index_relation::asameb;
    };

  while (inew < num_new && iold < num_old) {
    index_relation relation = compare_indexes(iset_new[inew], iset_old[iold]);

    if (relation == index_relation::asameb) {
      // Point survived adaptation: Transfer data
      transfers.push_back(iold);
      transfers.push_back(inew);
      inew++;
      iold++;
    } else if (relation == index_relation::abeforeb) {
      // New point added (Refinement): Already 0.0, just advance
      inew++;
    } else {
      // Old point removed (Coarsening): Discard data
      iold++;
    }
  }

  // Peform copy on GPU
  gpu::vector<int64_t> gpu_transfers(transfers);
  gpu::flagged_memcopy_dev2dev(block_size, gpu_transfers, x, x_new);
  std::swap(x, x_new);
}

template<typename P>
void poisson_md<P>::solve_potential_(gpu::vector<P> &density, sparse_grid const &position_grid,
                                     connection_patterns const &conn, kronmult::workspace<P> &work, poisson_bc const bc)
{
  // Remap the previous potential to the new grid if needed to use as a warm start
  if (generation != position_grid.generation()) {
    remap_(iset_, position_grid.iset(), gpu_potential);
    iset_ = position_grid.iset();
    generation = position_grid.generation();
  }

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
  int num_iter = 0;
  if (precon.method() == precon_method::jacobi) {
    num_iter = cg_solver.solve([&](P y[]) -> void
      {
        tools::time_event timing_("poisson_md jacobi preconditioner");
        gpu::jacobi_apply(precon.gpu_jacobi(), y);
      }, apply_lhs, density, gpu_potential);
  } else {
    num_iter = cg_solver.solve(nullptr, apply_lhs, density, gpu_potential);
  }
  // std::cout << num_iter << " iterations\n";
  std::ignore = num_iter;

  // Restore the density
  if (bc == poisson_bc::periodic)
    gpu::memcopy_dev2dev(1, gpu_density0.data(), density.data());
}
#endif

template<typename P>
void poisson_md<P>::kron_diag_(term_entry<P> const &tme, sparse_grid const &grid, connection_patterns const &conn,
                              int const block_size, std::vector<P> &y) const
{
#pragma omp parallel
  {
    std::array<P const *, max_num_dimensions> amats;

#pragma omp for
    for (int i = 0; i < grid.num_indexes(); i++) {
      for (int d : iindexof(num_dims))
        amats[d] = tme.coeffs[d][conn[tme.coeffs[d]].row_diag(grid[i][d])];

      for (int t : iindexof(block_size)) {
        P a = 1;
        int tt = t;
        for (int d = num_dims - 1; d >= 0; --d)
        {
          if (amats[d] != nullptr) {
            int const rc = tt % pdof;
            a *= amats[d][rc * pdof + rc];
          }
          tt /= pdof;
        }
        y[i * block_size + t] += a;
      }
    }
  }
}

template<typename P>
void poisson_md<P>::update_preconditioner(sparse_grid const &position_grid, connection_patterns const &conn,
                                          poisson_bc const bc)
{
  // return if nothing more to do
  if (precon.valid_for(position_grid))
    return;

  tools::time_event timing_("updating poisson_md preconditioner");

  precon.grid_gen = position_grid.generation(); // update the grid gen

  precon_method const method = precon; // get the precon method
  int const block_size = fm::ipow(pdof, position_grid.num_dims());
  int64_t const num_entries = block_size * position_grid.num_indexes();

  if (method == precon_method::jacobi) {
    std::vector<P> &jacobi = precon.jacobi();

    if (jacobi.size() == 0)
      jacobi.resize(num_entries);
    else {
      jacobi.resize(num_entries);
      std::fill(jacobi.begin(), jacobi.end(), P{0});
    }

    for (term_entry<P> const &tentry : laplacian_terms)
      kron_diag_(tentry, position_grid, conn, block_size, jacobi);

    if (bc == poisson_bc::periodic)
      jacobi[0] = P{0};
    else
      jacobi[0] = P{1} / jacobi[0];
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 1; i < jacobi.size(); i++)
      jacobi[i] = P{1} / jacobi[i];

    #ifdef ASGARD_USE_GPU
    compute->set_device(gpu::device{0});
    precon.gpu_jacobi() = jacobi;
    #endif
  }
}

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