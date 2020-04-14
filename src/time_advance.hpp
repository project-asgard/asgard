#pragma once
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "timer.hpp"

// this function executes a time step using the current solution
// vector x (in host_space).
// on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_time_advance(PDE<P> const &pde, element_table const &table,
                      std::vector<fk::vector<P>> const &unscaled_sources,
                      std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                      fk::vector<P> const &x,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, P const time, P const dt);

template<typename P>
fk::vector<P>
implicit_time_advance(PDE<P> const &pde, element_table const &table,
                      std::vector<fk::vector<P>> const &unscaled_sources,
                      std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                      fk::vector<P> const &x,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, P const time, P const dt,
                      solve_opts const solver = solve_opts::direct,
                      bool update_system      = true);

// apply the system matrix to the current solution vector using batched
// gemm.
template<typename P>
static fk::vector<P>
apply_A(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &grid, std::vector<element_chunk> const &chunks,
        fk::vector<P> const &x)
{
  fk::vector<P> fx(x.size());
  batch_workspace<P, resource::device> batch_space(pde, grid, chunks);

  for (auto const &chunk : chunks)
  {
    // copy inputs onto GPU
    batch_space.input.transfer_from(x);

    // build batches for this chunk
    auto const batch_id = timer::record.start("build_batches");
    batch_chain<P, resource::device, chain_method::advance> const batches(
        pde, elem_table, batch_space, grid, chunk);
    timer::record.stop(batch_id);

    // execute
    auto const gemm_id = timer::record.start("batched_gemm");
    batches.execute();
    timer::record.stop(gemm_id);

    // do the reduction
    auto const reduce_id = timer::record.start("reduce_chunk");
    reduce_chunk(pde, batch_space.reduction_space, batch_space.output,
                 batch_space.get_unit_vector(), grid, chunk);
    timer::record.stop(reduce_id);
  }

  // copy outputs back from GPU
  fx.transfer_from(batch_space.output);
  return fx;
}
