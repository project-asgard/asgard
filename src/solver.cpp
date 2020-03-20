#include "solver.hpp"
#include "time_advance.hpp"

namespace solver
{
// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance)
{
  assert(tolerance >= std::numeric_limits<P>::epsilon());
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
  assert(max_iter <= n);

  P const norm_b = [&b]() {
    P const norm_b = fm::nrm2(b);
    if (norm_b == 0.0)
    {
      return static_cast<P>(1.0);
    }
    return norm_b;
  }();

  fk::vector<P> residual(b);
  auto const compute_residual = [&A, &x, &b, &norm_b, &residual, &do_precond,
                                 &precond, &precond_pivots]() {
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

  auto const done = [](P const error, int const outer_iters,
                       int const inner_iters) {
    std::cout << "GMRES complete with error: " << error << '\n';
    std::cout << outer_iters << " outer iterations, " << inner_iters
              << " inner iterations\n";
  };

  P const norm_r = compute_residual();
  P error        = norm_r / norm_b;
  if (error < tolerance)
  {
    done(error, 0, 0);
    return error;
  }

  fk::matrix<P> basis(n, restart + 1);
  fk::matrix<P> krylov_proj(restart + 1, restart);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);

  int it = 0;
  int i  = 0;
  for (; it < max_iter; ++it)
  {
    P const norm_r = compute_residual();

    basis.update_col(0, residual * (1 / norm_r));
    // TODO what is s??
    fk::vector<P> s(n + 1);
    s(0) = norm_r;
    for (i = 0; i < restart; ++i)
    {
      fk::vector<P> new_basis =
          A * fk::vector<P, mem_type::view>(basis, i, 0, basis.nrows() - 1);

      if (do_precond)
      {
        fm::getrs(precond, new_basis, precond_pivots);
      }
      for (int k = 0; k <= i; ++k)
      {
        fk::vector<P, mem_type::const_view> const basis_vect(basis, k, 0,
                                                             basis.nrows() - 1);
        krylov_proj(k, i) = new_basis * basis_vect;
        new_basis         = new_basis - (basis_vect * krylov_proj(k, i));
      }
      krylov_proj(i + 1, i) = fm::nrm2(new_basis);

      basis.update_col(i + 1, new_basis * (1 / krylov_proj(i + 1, i)));
      for (int k = 0; k <= i - 1; ++k)
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

      krylov_proj(i + 1, i) = 0.0;
      P const temp          = cosines(i) * s(i);
      s(i + 1)              = -sines(i) * s(i);
      s(i)                  = temp;
      error                 = std::abs(s(i + 1)) / norm_b;

      if (error <= tolerance)
      {
        auto const proj =
            fk::matrix<P, mem_type::view>(krylov_proj, 0, i, 0, i);
        std::vector<int> pivots(i + 1);
        // TODO what is this "s"
        auto s_view = fk::vector<P, mem_type::view>(s, 0, i);
        fm::gesv(proj, s_view, pivots);
        x = x +
            (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0, i) *
             s_view);
        break; // depart the inner iteration loop
      }
    } // end of inner iteration loop

    if (error <= tolerance)
    {
      done(error, it, i);
      return error; // all done!
    }
    auto const proj = fk::matrix<P, mem_type::view>(krylov_proj, 0, restart - 1,
                                                    0, restart - 1);
    auto s_view     = fk::vector<P, mem_type::view>(s, 0, restart - 1);
    std::vector<int> pivots(restart);
    fm::gesv(proj, s_view, pivots);
    x = x + (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0,
                                           restart - 1) *
             s_view);
    P const norm_r_outer             = compute_residual();
    s(std::min(s.size() - 1, i + 1)) = norm_r_outer;
    error                            = norm_r_outer / norm_b;

    if (error <= tolerance)
    {
      done(error, it, i);
      return error;
    }
  } // end outer iteration

  done(error, it, i);
  return error;
}

template float simple_gmres(fk::matrix<float> const &A, fk::vector<float> &x,
                            fk::vector<float> const &b,
                            fk::matrix<float> const &M, int const restart,
                            int const max_iter, float const tolerance);

template double simple_gmres(fk::matrix<double> const &A, fk::vector<double> &x,
                             fk::vector<double> const &b,
                             fk::matrix<double> const &M, int const restart,
                             int const max_iter, double const tolerance);

} // namespace solver
