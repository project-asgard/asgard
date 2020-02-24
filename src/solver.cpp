#include "solver.hpp"

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
  auto const system_size = elem_size * static_cast<int64_t>(table.size());
  assert(restart < system_size);

  assert(threshold > 0.0);
  assert(theshold < 1.0);

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
}

// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance)
{
  assert(A.nrows() == A.ncols());
  int const n = A.nrows();
  assert(b.size() == n);
  assert(x.size() == n);

  bool const precond = M.size() > 0;
  std::vector<int> precond_pivots;
  if (precond)
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
    return 1.0;
  }();

  fk::vector<P> residual(b);
  auto const
      compute_residual[&A, &x, &b, &residual, &precond, &precond_pivots, ]()
  {
    static bool factored = false;
    bool const trans_A   = false;
    P const alpha        = -1.0;
    P const beta         = 1.0;
    fm::gemv(A, x, residual, trans_A, alpha, beta);
    if (precond)
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
    return fm::norm2(residual);
  };

  P const norm_r        = compute_residual();
  P const initial_error = norm_r / norm_b;
  if (initial_error < tolerance)
  {
    return initial_error;
  }

  fk::matrix<P> basis(n, restart);
  fk::matrix<P> krylov_proj(restart + 1, restart);
  fk::vector<P> sines(restart);
  fk::vector<P> cosines(restart);

  for (int i = 0; i < max_iter; ++i)
  {
    P const norm_r = compute_residual();

    basis.update_col(0, residual * (1 / norm_r));
    fk::vector<P> s(n);
    s(0) = norm_r;

    for (int j = 0; j < restart; ++j)
    {
      // FIXME need vector view from matrix row/column
      fk::vector<P> w = A * basis.extract
    }
  }
}

template float
gmres(PDE<float> const &pde, element_table const &elem_table,
      distribution_plan const &plan, std::vector<element_chunk> const &chunks,
      host_workspace<float> &host_space, rank_workspace<float> &rank_space,
      std::vector<batch_operands_set<float>> &batches, float const dt);

template double
gmres(PDE<double> const &pde, element_table const &elem_table,
      std::vector<element_chunk> const &chunks, distribution_plan const &plan,
      host_workspace<double> &host_space, rank_workspace<double> &rank_space,
      std::vector<batch_operands_set<double>> &batches, double const dt);

template float float
simple_gmres(fk::matrix<float> const &A, fk::vector<float> const &x,
             fk::vector<float> &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

template double double
simple_gmres(fk::matrix<double> const &A, fk::vector<double> const &x,
             fk::vector<double> &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);
