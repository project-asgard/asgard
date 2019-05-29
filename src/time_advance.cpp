#include "time_advance.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, fk::vector<P> &x,
                           fk::vector<P> &x_orig, fk::vector<P> &fx,
                           fk::vector<P> &scaled_source,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           std::vector<fk::vector<P>> &workspace,
                           std::vector<batch_operands_set<P>> const &batches,
                           P const time, P const dt)
{
  assert(scaled_source.size() == x.size());
  assert(x.size() == fx.size());
  assert(x_orig.size() == x.size());
  fk::copy(x, x_orig);

  assert(workspace.size() == 3);
  for (fk::vector<P> &vect : workspace)
  {
    assert(vect.size() == x.size());
  }
  assert(time >= 0);

  assert(static_cast<int>(batches.size()) == pde.num_dims + 1);
  for (batch_operands_set<P> const &ops : batches)
  {
    assert(ops.size() == 3);
  }
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);

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

  P const alpha = 1.0;
  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, scaled_source, time);
  fk::axpy(alpha, scaled_source, fx);
  fk::copy(fx, workspace[0]);
  P const fx_scale_1 = a21 * dt;
  fk::axpy(fx_scale_1, fx, x);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, scaled_source, time + c2 * dt);
  fk::axpy(alpha, scaled_source, fx);
  fk::copy(fx, workspace[1]);
  fk::copy(x_orig, x);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;
  fk::axpy(fx_scale_2a, workspace[0], x);
  fk::axpy(fx_scale_2b, workspace[1], x);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, scaled_source, time + c3 * dt);
  fk::axpy(alpha, scaled_source, fx);
  fk::copy(fx, workspace[2]);

  P const scale_0 = dt * b1;
  P const scale_1 = dt * b2;
  P const scale_2 = dt * b3;
  fk::scal(static_cast<P>(0.0), fx);
  fk::copy(x_orig, x);
  fk::axpy(scale_0, workspace[0], x);
  fk::axpy(scale_1, workspace[1], x);
  fk::axpy(scale_2, workspace[2], x);

  fk::copy(x, fx);
}

// scale source vectors for time
template<typename P>
static fk::vector<P> &
scale_sources(PDE<P> const &pde,
              std::vector<fk::vector<P>> const &unscaled_sources,
              fk::vector<P> &scaled_source, P const time)
{
  // zero out final vect
  fk::scal(static_cast<P>(0.0), scaled_source);
  // scale and accumulate all sources
  for (int i = 0; i < pde.num_sources; ++i)
  {
    fk::axpy(pde.sources[i].time_func(time), unscaled_sources[i],
             scaled_source);
  }
  return scaled_source;
}

// apply the system matrix to the current solution vector using batched
// gemm (explicit time advance).
template<typename P>
static void apply_explicit(std::vector<batch_operands_set<P>> const &batches)
{
  // batched gemm
  P const alpha = 1.0;
  P const beta  = 0.0;
  for (int i = 0; i < static_cast<int>(batches.size()) - 1; ++i)
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
explicit_time_advance(PDE<float> const &pde, fk::vector<float> &x,
                      fk::vector<float> &x_orig, fk::vector<float> &fx,
                      fk::vector<float> &scaled_source,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      std::vector<fk::vector<float>> &workspace,

                      std::vector<batch_operands_set<float>> const &batches,
                      float const time, float const dt);

template void
explicit_time_advance(PDE<double> const &pde, fk::vector<double> &x,
                      fk::vector<double> &x_orig, fk::vector<double> &fx,
                      fk::vector<double> &scaled_source,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      std::vector<fk::vector<double>> &workspace,
                      std::vector<batch_operands_set<double>> const &batches,
                      double const time, double const dt);
