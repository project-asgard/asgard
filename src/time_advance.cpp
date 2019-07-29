#include "time_advance.hpp"
#include "element_table.hpp"
#include "fast_math.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           rank_workspace<P> &rank_space,
                           std::vector<element_chunk> chunks, P const time,
                           P const dt)
{
  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);

  fm::copy(host_space.x, host_space.x_orig);
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

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source, time);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_1);
  P const fx_scale_1 = a21 * dt;
  fm::axpy(host_space.fx, host_space.x, fx_scale_1);

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c2 * dt);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_2);
  fm::copy(host_space.x_orig, host_space.x);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;
  fm::axpy(host_space.result_1, host_space.x, fx_scale_2a);
  fm::axpy(host_space.result_2, host_space.x, fx_scale_2b);

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c3 * dt);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_3);

  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::copy(host_space.x_orig, host_space.x);
  fm::axpy(host_space.result_1, host_space.x, scale_1);
  fm::axpy(host_space.result_2, host_space.x, scale_2);
  fm::axpy(host_space.result_3, host_space.x, scale_3);

  fm::copy(host_space.x, host_space.fx);
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
static void
apply_explicit(PDE<P> const &pde, element_table const &elem_table,
               std::vector<element_chunk> const &chunks,
               host_workspace<P> &host_space, rank_workspace<P> &rank_space)
{
  fm::scal(static_cast<P>(0.0), host_space.fx);
  for (auto const &chunk : chunks)
  {
    // copy in inputs
    copy_chunk_inputs(pde, rank_space, host_space, chunk);

    // build batches for this chunk
    std::vector<batch_operands_set<P>> batches =
        build_batches(pde, elem_table, rank_space, chunk);

    // do the gemms
    P const alpha = 1.0;
    P const beta  = 0.0;
    for (int i = 0; i < pde.num_dims; ++i)
    {
      batch<P> const a = batches[i][0];
      batch<P> const b = batches[i][1];
      batch<P> const c = batches[i][2];

      batched_gemm(a, b, c, alpha, beta, resource::device);
    }

    // do the reduction
    reduce_chunk(pde, rank_space, chunk);

    // copy outputs back
    copy_chunk_outputs(pde, rank_space, host_space, chunk);
  }
}

template void
explicit_time_advance(PDE<float> const &pde, element_table const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_chunk> chunks, float const time,
                      float const dt);

template void
explicit_time_advance(PDE<double> const &pde, element_table const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double> &host_space,
                      rank_workspace<double> &rank_space,
                      std::vector<element_chunk> chunks, double const time,
                      double const dt);
