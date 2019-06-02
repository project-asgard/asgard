#include "time_advance.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           explicit_workspace<P> &work,
                           work_set<P> const &batches, P const time, P const dt)
{
  assert(work.scaled_source.size() == system.x.size());
  assert(system.x.size() == system.fx.size());
  assert(work.x_orig.size() == system.x.size());
  assert(work.result_1.size() == system.x.size());
  assert(work.result_2.size() == system.x.size());
  assert(work.result_3.size() == system.x.size());

  fk::copy(system.x, work.x_orig);

  assert(time >= 0);

  for (auto const &ops_list : batches)
  {
    assert(static_cast<int>(ops_list.size()) == pde.num_dims + 1);
    for (batch_operands_set<P> ops : ops_list)
    {
      assert(ops.size() == 3);
    }
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
  scale_sources(pde, unscaled_sources, work.scaled_source, time);
  fk::axpy(alpha, work.scaled_source, system.fx);
  fk::copy(system.fx, work.result_1);
  P const fx_scale_1 = a21 * dt;
  fk::axpy(fx_scale_1, system.fx, system.x);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, work.scaled_source, time + c2 * dt);
  fk::axpy(alpha, work.scaled_source, system.fx);
  fk::copy(system.fx, work.result_2);
  fk::copy(work.x_orig, system.x);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;
  fk::axpy(fx_scale_2a, work.result_1, system.x);
  fk::axpy(fx_scale_2b, work.result_2, system.x);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, work.scaled_source, time + c3 * dt);
  fk::axpy(alpha, work.scaled_source, system.fx);
  fk::copy(system.fx, work.result_3);

  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fk::copy(work.x_orig, system.x);
  fk::axpy(scale_1, work.result_1, system.x);
  fk::axpy(scale_2, work.result_2, system.x);
  fk::axpy(scale_3, work.result_3, system.x);

  fk::copy(system.x, system.fx);
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
static void apply_explicit(work_set<P> const &batches)
{
  // batched gemm
  P const alpha = 1.0;
  P const beta  = 0.0;

  for (int i = 0; i < static_cast<int>(batches.size()); ++i)
  {
    auto const batch_operands_list = batches[i];
    for (int j = 0; j < static_cast<int>(batch_operands_list.size()) - 1; ++j)
    {
      batch<P> const a = batch_operands_list[j][0];
      batch<P> const b = batch_operands_list[j][1];
      batch<P> const c = batch_operands_list[j][2];
      batched_gemm(a, b, c, alpha, beta);
    }

    // reduce
    batch<P> const r_a = batch_operands_list[batch_operands_list.size() - 1][0];
    batch<P> const r_b = batch_operands_list[batch_operands_list.size() - 1][1];
    batch<P> const r_c = batch_operands_list[batch_operands_list.size() - 1][2];
    P const reduction_beta = (i == 0) ? 0.0 : 1.0;
    batched_gemv(r_a, r_b, r_c, alpha, reduction_beta);
  }
}

template void
explicit_time_advance(PDE<float> const &pde,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      explicit_system<float> &system,
                      explicit_workspace<float> &work,
                      work_set<float> const &batches, float const time,
                      float const dt);
template void
explicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      explicit_workspace<double> &work,
                      work_set<double> const &batches, double const time,
                      double const dt);
