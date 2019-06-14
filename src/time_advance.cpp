#include "time_advance.hpp"
#include "fast_math.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           work_set<P> const &batches, P const time, P const dt)
{
  assert(system.scaled_source.size() == system.batch_input.size());
  assert(system.batch_input.size() == system.batch_output.size());
  assert(system.x_orig.size() == system.batch_input.size());
  assert(system.result_1.size() == system.batch_input.size());
  assert(system.result_2.size() == system.batch_input.size());
  assert(system.result_3.size() == system.batch_input.size());

  fm::copy(system.batch_input, system.x_orig);

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

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, system.scaled_source, time);
  fm::axpy(system.scaled_source, system.batch_output);
  fm::copy(system.batch_output, system.result_1);
  P const fx_scale_1 = a21 * dt;
  fm::axpy(system.batch_output, system.batch_input, fx_scale_1);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, system.scaled_source, time + c2 * dt);
  fm::axpy(system.scaled_source, system.batch_output);
  fm::copy(system.batch_output, system.result_2);
  fm::copy(system.x_orig, system.batch_input);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;
  fm::axpy(system.result_1, system.batch_input, fx_scale_2a);
  fm::axpy(system.result_2, system.batch_input, fx_scale_2b);

  apply_explicit(batches);
  scale_sources(pde, unscaled_sources, system.scaled_source, time + c3 * dt);
  fm::axpy(system.scaled_source, system.batch_output);
  fm::copy(system.batch_output, system.result_3);

  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::copy(system.x_orig, system.batch_input);
  fm::axpy(system.result_1, system.batch_input, scale_1);
  fm::axpy(system.result_2, system.batch_input, scale_2);
  fm::axpy(system.result_3, system.batch_input, scale_3);

  fm::copy(system.batch_input, system.batch_output);
}

// scale source vectors for time
template<typename P>
static fk::vector<P> &
scale_sources(PDE<P> const &pde,
              std::vector<fk::vector<P>> const &unscaled_sources,
              fk::vector<P> &scaled_source, P const time)
{
  // zero out final vect
  fm::scal(static_cast<P>(0.0), scaled_source);
  // scale and accumulate all sources
  for (int i = 0; i < pde.num_sources; ++i)
  {
    fm::axpy(unscaled_sources[i], scaled_source,
             pde.sources[i].time_func(time));
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
                      work_set<float> const &batches, float const time,
                      float const dt);
template void
explicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      work_set<double> const &batches, double const time,
                      double const dt);
