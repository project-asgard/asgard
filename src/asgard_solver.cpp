#include "asgard_solver.hpp"

#include "asgard_small_mats.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard::solvers
{

template<typename P>
void direct<P>::update(
    group_id group, size_t stage, sparse_grid const &grid, connection_patterns const &conn,
    term_manager<P> const &terms, P alpha)
{
  tools::time_event timing_("forming dense matrix");
  int const num_dims    = grid.num_dims();
  int const num_indexes = grid.num_indexes();
  int const pdof        = terms.basis.pdof;

  int const n = fm::ipow(pdof, num_dims);

  block_matrix<P> bmat(n * n, num_indexes, num_indexes);
  block_matrix<P> wmat(n * n, num_indexes, num_indexes);

  std::array<block_matrix<P>, max_num_dimensions> ids; // identity coefficients
  for (int d : iindexof(num_dims)) {
    int const size = fm::ipow2(grid.current_level(d));
    ids[d] = block_matrix<P>(pdof * pdof, size, size);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < pdof; j++)
        ids[d](i, i)[j * pdof + j] = 1;
    }
  }

  // work coefficients, full-block matrices that will be used for this term_md
  std::array<block_matrix<P> const *, max_num_dimensions> wcoeffs;

  std::array<block_matrix<P>, max_num_dimensions> temp_mats;

  auto kron_mats = [&](block_matrix<P> &mat)
      -> void
    {
      if (num_dims == 1) {
#pragma omp parallel for
        for (int c = 0; c < num_indexes; c++) {
          for (int r = 0; r < num_indexes; r++) {
            int const ic = grid[c][0];
            int const ir = grid[r][0];

            std::copy_n((*wcoeffs[0])(ir, ic), pdof * pdof, mat(r, c));
          }
        }
      } else {
#pragma omp parallel for
        for (int c = 0; c < num_indexes; c++) {
          for (int r = 0; r < num_indexes; r++) {
            int const *ic = grid[c];
            int const *ir = grid[r];

            int cyc    = 1;
            int stride = fm::ipow(pdof, num_dims - 1);
            int repeat = stride;
            for (int d : iindexof(num_dims)) {
              smmat::kron_block(pdof, cyc, stride, repeat,
                                (*wcoeffs[d])(ir[d], ic[d]), mat(r, c));
              stride /= pdof;
              cyc    *= pdof;
            }
          }
        }
      }
    };

  auto set_wcoeff = [&](term_entry<P> const &te)
      -> void
    {
      for (int d : iindexof(num_dims)) {
        if (te.coeffs[d].nblock() > 0) {
          temp_mats[d] = te.coeffs[d].to_full(conn);
          wcoeffs[d]   = &temp_mats[d];
        } else {
          wcoeffs[d] = &ids[d];
        }
      }
    };

  indexrange trange = terms.terms_group_range(group);

  int icurrent = trange.ibegin();
  while (icurrent < trange.iend())
  {
    auto it = terms.terms.begin() + icurrent;

    #ifdef ASGARD_USE_MPI
    if (not terms.resources.owns(it->rec)) {
      icurrent += it->num_chain;
      continue;
    }
    #endif

    if (it->num_chain == 1) {
      set_wcoeff(*it);
      wmat.fill(1);
      kron_mats(wmat);

      int64_t const size = n * n * num_indexes * num_indexes;
      P *mat_data        = bmat.data();
      P const *wmat_data = wmat.data();

      ASGARD_OMP_PARFOR_SIMD
      for (int64_t i = 0; i < size; i++)
        mat_data[i] += wmat_data[i];

      ++icurrent;
    } else {
      if (it->num_chain == 2) {
        // need two temp matrices
        block_matrix<P> t1(n * n, num_indexes, num_indexes);
        set_wcoeff(*it);
        t1.fill(1);
        kron_mats(t1);

        block_matrix<P> t2(n * n, num_indexes, num_indexes);
        set_wcoeff(*(it + 1));
        t2.fill(1);
        kron_mats(t2);

        gemm1(n, t1, t2, bmat);
      } else {
        throw std::runtime_error(
            "term_md chains with num_chain >= 3 are not yet implemented "
            "for the direct solver");
      }

      icurrent += it->num_chain;
    }
  }

  size_t const idx = mat_index(group, stage); // index for the matrix entry
  if (mats.size() <= idx)
    mats.resize(idx + 1);

  // must happen before the potential MPI return down below
  mats[idx].grid_gen = grid.generation();

  dense_matrix<P> &dense_mat = mats[idx].dense_mat;
  dense_mat = bmat.to_dense_matrix(n);

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (terms.resources.is_leader()) {
      dense_matrix<P> mat = dense_mat;
      terms.resources.reduce_add(static_cast<int>(dense_mat.nrows() * dense_mat.ncols()),
                                 mat.data(), dense_mat.data());
    } else {
      terms.resources.reduce_add(static_cast<int>(dense_mat.nrows() * dense_mat.ncols()),
                                 dense_mat.data());
      return;
    }
  }
  #endif

  if (alpha != 0)
  {
    int64_t const size = n * num_indexes;

    #pragma omp parallel for
    for (int64_t c = 0; c < size - 1; c++) {
      P *dd = dense_mat.data() + c * (size + 1);
      dd[0] = P{1} + alpha * dd[0];
      dd += 1;
      ASGARD_OMP_SIMD
      for (int64_t i = 0; i < size; i++) {
        dd[i] *= alpha;
      }
    }

    dense_mat(size - 1, size - 1) = P{1} + alpha * dense_mat(size - 1, size - 1);
  }

  dense_mat.factorize();
}

template<typename P>
size_t direct<P>::used_bytes() const {
  size_t t = 0;
  for (auto const &m : mats) t += m.dense_mat.used_bytes();
  return t;
}

template<typename P>
void scaled_identity<P>::update(group_id group, size_t stage, sparse_grid const &grid,
                                term_manager<P> const &terms, P alpha)
{
  #ifdef ASGARD_USE_GPU
  num_entries = grid.num_indexes() * fm::ipow(terms.basis.pdof, grid.num_dims());
  #endif

  int const num_dims = terms.grid.num_dims();

  indexrange trange = terms.terms_group_range(group);

  if (grid_gen(group, stage) == -1) {
    // first time getting a call here, verify that the operators are scaled identity
    for (int i : trange) {
      term_md<P> const &term = terms.terms[i].tmd;
      rassert(term.is_separable(), "non-separable term detected in the scaled-identity solver");
      for (int d : iindexof(num_dims)) {
        term_1d<P> const &t1d = term.dim(d);
        rassert(t1d.is_identity() or t1d.is_volume(),
                "scaled-identity solver can be used only with volume and identity instances of term1d");
        if (t1d.is_volume())
          rassert(not t1d.rhs(), "detected non-constant coefficient for the scaled-identity solver");
      }
    }
  }

  P scal = 1;
  for (int i : trange) {
    term_md<P> const &term = terms.terms[i].tmd;
    assert(term.is_separable() and term.flux_dim() == -1);
    for (int d : iindexof(num_dims)) {
      term_1d<P> const &t1d = term.dim(d);
      if (t1d.is_volume())
        scal *= t1d.rhs_const();
    }
  }

  if (alpha != 0)
    set_alpha(group, stage, P{1} + alpha * scal); // Euler I + alpha * nu * I
  else
    set_alpha(group, stage, scal); // steady state, nu * I

  size_t const idx = s_index(group, stage);
  scale_[idx].grid_gen = grid.generation();
}

template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, std::vector<P> &x) const
{
  P const s = scale(group, stage);
  ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    x[i] = s * x[i];
}

#ifdef ASGARD_USE_GPU
template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, gpu::vector<P> &x) const
{
  assert(num_entries == x.size());
  gpu::set_scal(x.size(), scale(group, stage), x.data());
}
template<typename P>
void scaled_identity<P>::operator()(group_id group, size_t stage, P x[]) const
{
  gpu::set_scal(num_entries, scale(group, stage), x);
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class direct<double>;
template class scaled_identity<double>;
#endif // ASGARD_ENABLE_DOUBLE

#ifdef ASGARD_ENABLE_FLOAT
template class direct<float>;
template class scaled_identity<float>;
#endif // ASGARD_ENABLE_FLOAT

} // namespace asgard::solvers

namespace asgard
{

template<typename P>
void solver_manager<P>::update_grid(
    group_id group, size_t stage, sparse_grid const &grid,
    connection_patterns const &conn, term_manager<P> const &terms, P alpha,
    preconditioner_data<P> &precon)
{
  tools::time_event timing_("updating solver");
  // assuming that the moments will cause the terms to change all the time
  // therefore, we need to update the matrices and preconditioners
  bool const needs_update = terms.has_sep_moments(group);

  // first, update the solver itself
  switch (method()) {
    case solver_method::direct: {
      solvers::direct<P> &solver = std::get<solvers::direct<P>>(var);
      if (needs_update or solver.grid_gen(group, stage) != grid.generation())
        solver.update(group, stage, grid, conn, terms, alpha);
    }
    break;
    case solver_method::scaled_identity: {
      solvers::scaled_identity<P> &solver = std::get<solvers::scaled_identity<P>>(var);
      if (needs_update or solver.grid_gen(group, stage) != grid.generation())
        solver.update(group, stage, grid, terms, alpha);
    }
    break;
    default: // iterative solvers don't need updating
      break;
  };

  // second, update the preconditioner
  // return if nothing more to do
  if (not needs_update and precon.valid_for(grid))
    return;

  precon.grid_gen = grid.generation(); // update the grid gen

  precon_method const method = precon; // get the precon method

  if (method == precon_method::jacobi) {
    std::vector<P> &jacobi = precon.jacobi();

    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      if (terms.resources.is_leader()) {
        terms.make_jacobi(group, terms.mpiwork);
        terms.resources.reduce_add(terms.mpiwork, jacobi);
      } else {
        terms.make_jacobi(group, jacobi);
        terms.resources.reduce_add(jacobi);
        return;
      }
    } else {
      terms.make_jacobi(group, jacobi);
    }
    #else
    terms.make_jacobi(group, jacobi);
    #endif

    if (alpha == 0) { // steady state solver
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < jacobi.size(); i++)
        jacobi[i] = P{1} / jacobi[i];
    } else {
      ASGARD_OMP_PARFOR_SIMD
      for (size_t i = 0; i < jacobi.size(); i++)
        jacobi[i] = P{1} / (P{1} + alpha * jacobi[i]);
    }

    #ifdef ASGARD_USE_GPU
    if (terms.resources.is_leader()) {
      compute->set_device(gpu::device{0});
      precon.gpu_jacobi() = jacobi;
    }
    #endif
  }
}

template<typename P>
void solver_manager<P>::xpby(std::vector<P> const &x, P beta, P y[]) {
ASGARD_OMP_PARFOR_SIMD
  for (size_t i = 0; i < x.size(); i++)
    y[i] = x[i] + beta * y[i];
}

#ifdef ASGARD_USE_GPU
template<typename P>
void solver_manager<P>::iterate_solve(
    solvers::operation_apply_precon<P> prec, solvers::operation_apply_lhs<P> apply_lhs,
    gpu::vector<P> const &rhs, gpu::vector<P> &x) const
{
  if (method() == solver_method::bicgstab) {
    if (prec) {
      solvers::bicgstab<P> const &bicg = std::get<solvers::bicgstab<P>>(var);

      bicg.prec_y_gpu.resize(rhs.size());

      bicg.prec_rhs_gpu = rhs;
      prec(bicg.prec_rhs_gpu.data());

      num_apply += bicg.solve([&](P alpha, P const xx[], P beta, P y[])
          -> void {
            if (beta == 0) {
              apply_lhs(alpha, xx, 0, y);
              prec(y);
            } else {
              apply_lhs(alpha, xx, 0, bicg.prec_y_gpu.data());
              prec(bicg.prec_y_gpu.data());
              gpu::xpby(bicg.prec_y_gpu, beta, y);
            }
          }, bicg.prec_rhs_gpu, x);
    } else {
      num_apply += std::get<solvers::bicgstab<P>>(var).solve(apply_lhs, rhs, x);
    }
  } else { // if (opt == solve_opts::gmres)
    if (prec) {
      num_apply += std::get<solvers::gmres<P>>(var).solve(prec, apply_lhs, rhs, x);
    } else {
      num_apply += std::get<solvers::gmres<P>>(var).solve(
        [](P *)->void{ /* no preconditioner */ }, apply_lhs, rhs, x);
    }
  }
}
#endif

template<typename P>
void solver_manager<P>::print_opts(std::ostream &os) const
{
  os << "solver:\n";
  switch (method()) {
    case solver_method::direct:
      os << "  direct\n";
      break;
    case solver_method::bicgstab:
      os << "  bicgstab\n";
      os << "  tolerance:      " << std::get<solvers::bicgstab<P>>(var).tolerance() << '\n';
      os << "  max iterations: " << std::get<solvers::bicgstab<P>>(var).max_iter() << '\n';
      break;
    case solver_method::gmres:
      os << "  gmres\n";
      os << "  tolerance: " << std::get<solvers::gmres<P>>(var).tolerance() << '\n';
      os << "  max inner: " << std::get<solvers::gmres<P>>(var).max_inner() << '\n';
      os << "  max outer: " << std::get<solvers::gmres<P>>(var).max_outer() << '\n';
      break;
    case solver_method::scaled_identity:
      os << "  scaled-identity\n";
      break;
    default:
      break;
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template struct solver_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template struct solver_manager<float>;
#endif

}
