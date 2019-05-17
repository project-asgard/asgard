#include "time_advance.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, fk::vector<P> const &x,
                           fk::vector<P> const &fx,
                           std::vector<batch_operands_set<P>> const &batches,
                           std::vector<fk::vector<P>> &unscaled_sources,
                           fk::vector<P> &scaled_source, P const time)
{}

// scale source vectors for time
template<typename P>
static void scale_sources(std::vector<fk::vector<P>> &unscaled_source,
                          fk::vector<P> &scaled_source)
{}

// apply the system matrix to the current solution vector using batched
// gemm (explicit time advance).
template<typename P>
static void apply_explicit(std::vector<batch_operands_set<P>> const &batches)
{
  // batched gemm
  P const alpha = 1.0;
  P const beta  = 0.0;
  for (int i = 0; i < batches.size() - 1; ++i)
  {
    batch<P> const a = batches[i][0];
    batch<P> const b = batches[i][1];
    batch<P> const c = batches[i][2];
    batched_gemm(a, b, c, alpha, beta);
  }

  // reduce
  batch<P> const r_a = batches[batches.size() - 1][0];
  batch<P> const r_b = batches[batches.size() - 1][1];
  batch<P> const r_c = batches[batches.size() - 1][2];
  batched_gemv(r_a, r_b, r_c, alpha, beta);
}

template void
explicit_time_advance(PDE<float> const &pde, fk::vector<float> const &x,
                      fk::vector<float> const &fx,
                      std::vector<batch_operands_set<float>> const &batches,
                      std::vector<fk::vector<float>> &unscaled_sources,
                      fk::vector<float> &scaled_source, float const time);

template void
explicit_time_advance(PDE<double> const &pde, fk::vector<double> const &x,
                      fk::vector<double> const &fx,
                      std::vector<batch_operands_set<double>> const &batches,
                      std::vector<fk::vector<double>> &unscaled_sources,
                      fk::vector<double> &scaled_source, double const time);
