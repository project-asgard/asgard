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
void explicit_time_advance(
    PDE<P> const &pde, element_table const &table,
    std::vector<fk::vector<P>> const &unscaled_sources,
    std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
    host_workspace<P> &host_space, device_workspace<P> &dev_space,
    std::vector<element_chunk> const &chunks, distribution_plan const &plan,
    P const time, P const dt);

template<typename P>
void implicit_time_advance(
    PDE<P> const &pde, element_table const &table,
    std::vector<fk::vector<P>> const &unscaled_sources,
    std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
    host_workspace<P> &host_space, std::vector<element_chunk> const &chunks,
    distribution_plan const &plan, P const time, P const dt,
    solve_opts const solver = solve_opts::direct, bool update_system = true);

// apply the system matrix to the current solution vector using batched
// gemm.
template<typename P>
static void
apply_A(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &grid, std::vector<element_chunk> const &chunks,
        host_workspace<P> &host_space, device_workspace<P> &dev_space)
{
  fm::scal(static_cast<P>(0.0), host_space.fx);
  fm::scal(static_cast<P>(0.0), dev_space.batch_output);

  // copy inputs onto GPU
  dev_space.batch_input.transfer_from(host_space.x);

  for (auto const &chunk : chunks)
  {
    // build batches for this chunk
    auto const batch_id = timer::record.start("build_batches");
    batch_chain<P, resource::device, chain_method::advance> const batches(
        pde, elem_table, dev_space, grid, chunk);
    timer::record.stop(batch_id);

    // execute
    auto const gemm_id = timer::record.start("batched_gemm");
    batches.execute();
    timer::record.stop(gemm_id);

    // do the reduction
    auto const reduce_id = timer::record.start("reduce_chunk");
    reduce_chunk(pde, dev_space, grid, chunk);
    timer::record.stop(reduce_id);
  }

  // copy outputs back from GPU
  host_space.fx.transfer_from(dev_space.batch_output);
}
