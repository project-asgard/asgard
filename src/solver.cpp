#include "solver.hpp"
#include "time_advance.hpp"

namespace solver
{
template<typename P>
P gmres(PDE<P> const &pde, element_table const &elem_table,
        distribution_plan const &plan, std::vector<element_chunk> const &chunks,
        host_workspace<P> &host_space, rank_workspace<P> &rank_space,
        std::vector<batch_operands_set<P>> &batches, P const dt,
        P const threshold, int const restart)
{
  assert(restart > 0);
  auto const degree      = pde.get_dimensions()[0].get_degree();
  auto const elem_size   = static_cast<int>(std::pow(degree, pde.num_dims));
  auto const system_size = elem_size * static_cast<int64_t>(elem_table.size());
  assert(restart < system_size);

  assert(threshold > 0.0);
  assert(threshold < 1.0);

  int const my_rank = get_rank();

  // allocate space
  fk::matrix<P> basis(host_space.x.size(), restart);
  fk::matrix<P> subspace_projection(restart + 1, restart);
  fk::vector<P> cosines(restart);
  fk::vector<P> sines(restart);

  // initial residual -- may skip this eventually
  fm::copy(host_space.x, host_space.x_orig);
  apply_A(pde, elem_table, plan.at(my_rank), chunks, host_space, rank_space,
          batches);
  reduce_results(host_space.fx, host_space.reduced_fx, plan, my_rank);
  exchange_results(host_space.reduced_fx, host_space.x, elem_size, plan,
                   my_rank);
  return 0.0;
}

// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance)
{
  assert(A.nrows() == A.ncols());
  int const n = A.nrows();
  assert(b.size() == n);
  assert(x.size() == n);

  bool const do_precond = M.size() > 0;
  std::vector<int> precond_pivots;
  if (do_precond)
  {
    assert(M.ncols() == n);
    assert(M.nrows() == n);
    precond_pivots.resize(n);
  }
  fk::matrix<P> precond(M);

  assert(restart > 0);
  assert(restart <= n);
  assert(max_iter >= restart);

  P const norm_b = [&b]() {
    P const norm_b = fm::nrm2(b);
    if (norm_b > 0.0)
    {
      return norm_b;
    }
    return static_cast<P>(1.0);
  }();

  fk::vector<P> residual(b);
  auto const compute_residual = [&A, &x, &b, &residual, &do_precond, &precond,
                                 &precond_pivots]() {
    static bool factored = false;
    bool const trans_A   = false;
    P const alpha        = -1.0;
    P const beta         = 1.0;
    fm::gemv(A, x, residual, trans_A, alpha, beta);
    if (do_precond)
    {
      if (!factored)
      {
        fm::gesv(precond, residual, precond_pivots);
        factored = true;
      }
      else
      {
        fm::getrs(precond, residual, precond_pivots);
      }
    }
    return fm::nrm2(residual);
  };

  P const norm_r = compute_residual();
  P error        = norm_r / norm_b;
  if (error < tolerance)
  {
    return error;
  }

  fk::matrix<P> basis(n, restart);
  fk::matrix<P> krylov_proj(restart + 1, restart);
  fk::vector<P> sines(restart);
  fk::vector<P> cosines(restart);

  for (int i = 0; i < max_iter; ++i)
  {
    P const norm_r = compute_residual();

    basis.update_col(0, residual * (1 / norm_r));
    // TODO what is s??
    fk::vector<P> s(n);
    s(0) = norm_r;

    for (int j = 0; j < restart; ++j)
    {
      fk::vector<P> new_basis =
          A * fk::vector<P, mem_type::view>(basis, i, 0, basis.nrows() - 1);
      if (do_precond)
      {
        fm::getrs(precond, new_basis, precond_pivots);
      }
      for (int k = 0; k < j; ++k)
      {
        fk::vector<P, mem_type::const_view> const basis_vect(basis, k, 0,
                                                             basis.nrows()-1);
        krylov_proj(k, i) = new_basis * basis_vect;
        new_basis         = new_basis - basis_vect;
      }
      krylov_proj(i + 1, i) = fm::nrm2(new_basis);
      basis.update_col(i + 1, new_basis * (1 / krylov_proj(i + 1, i)));

      for (int k = 0; k < i - 1; ++k)
      {
        P const temp =
            cosines(k) * krylov_proj(k, i) + sines(k) * krylov_proj(k + 1, i);
        krylov_proj(k + 1, i) =
            -sines(k) * krylov_proj(k, i) + cosines(k) * krylov_proj(k + 1, i);
        krylov_proj(k, i) = temp;
      }

      // compute given's rotation
      lib_dispatch::rotg(krylov_proj.data(i, i), krylov_proj.data(i + 1, i),
                         cosines.data(i), sines.data(i));

      P const temp = cosines(i) * s(i);
      s(i + 1)     = -sines(i) * s(i);
      s(i)         = temp;

      krylov_proj(i, i) =
          cosines(i) * krylov_proj(i, i) + sines(i) * krylov_proj(i + 1, i);
      krylov_proj(i + 1, i) = 0.0;

      error = std::abs(s(i + 1)) / norm_b;
      if (error <= tolerance)
      {
        auto const proj =
            fk::matrix<P, mem_type::view>(krylov_proj, 0, i, 0, i);
        std::vector<int> pivots(i+1);
        // TODO what is this "s"
        auto s_view = fk::vector<P, mem_type::view>(s, 0, i);
        fm::gesv(proj, s_view, pivots);
        x = x +
            (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows()-1, 0, i) *
             s_view);
        break; // depart the inner iteration loop
      }
    } // end of inner iteration loop

    if (error <= tolerance)
      return error; // all done!

    auto proj   = krylov_proj.extract_submatrix(0, restart, 0, restart);
    auto s_view = fk::vector<P, mem_type::view>(s, 0, restart - 1);
    std::vector<int> pivots(i);
    fm::gesv(proj, s_view, pivots);
    x = x + (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows(), 0, i - 1) *
             s_view);
    P const norm_r_outer = compute_residual();
    s(i + 1)             = norm_r_outer;
    error                = norm_r_outer / norm_b;

    if (error <= tolerance)
      return error;
  } // end outer iteration
}

template float
gmres(PDE<float> const &pde, element_table const &elem_table,
      distribution_plan const &plan, std::vector<element_chunk> const &chunks,
      host_workspace<float> &host_space, rank_workspace<float> &rank_space,
      std::vector<batch_operands_set<float>> &batches, float const dt,
      float const threshold, int const restart);

template double
gmres(PDE<double> const &pde, element_table const &elem_table,
      distribution_plan const &plan, std::vector<element_chunk> const &chunks,
      host_workspace<double> &host_space, rank_workspace<double> &rank_space,
      std::vector<batch_operands_set<double>> &batches, double const dt,
      double const threshold, int const restart);

template float simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
                            fk::vector<float> const &b,
                            fk::matrix<float> const &M, int const restart,
                            int const max_iter, float const tolerance);

template double simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
                             fk::vector<double> const &b,
                             fk::matrix<double> const &M, int const restart,
                             int const max_iter, double const tolerance);

} // namespace solver
