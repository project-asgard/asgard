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
    host_workspace<P> &host_space, std::vector<element_chunk> const &chunks,
    distribution_plan const &plan, P const time, P const dt);

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
        host_workspace<P> &host_space)
{
  batch_workspace<P, resource::device> batch_space(pde, grid, chunks);
  fm::scal(static_cast<P>(0.0), host_space.fx);
  fm::scal(static_cast<P>(0.0), batch_space.output);

  for (auto const &chunk : chunks)
  {
<<<<<<< HEAD
    // build batches for this chunk
    auto const batch_id = timer::record.start("build_batches");
=======
    // copy inputs onto GPU
    batch_space.input.transfer_from(host_space.x);

    // build batches for this chunk
    auto const &batch_id = timer::record.start("build_batches");
>>>>>>> remove device workspace -> batch_workspace
    batch_chain<P, resource::device, chain_method::advance> const batches(
        pde, elem_table, batch_space, grid, chunk);
    timer::record.stop(batch_id);

    // execute
    auto const gemm_id = timer::record.start("batched_gemm");
    batches.execute();
    timer::record.stop(gemm_id);

    // do the reduction
<<<<<<< HEAD
    auto const reduce_id = timer::record.start("reduce_chunk");
    reduce_chunk(pde, dev_space, grid, chunk);
    timer::record.stop(reduce_id);
=======
    timer::record(reduce_chunk<P, resource::device>, "reduce_chunk", pde,
                  batch_space.reduction_space, batch_space.output,
                  batch_space.get_unit_vector(), grid, chunk);
>>>>>>> remove device workspace -> batch_workspace
  }

  // copy outputs back from GPU
  host_space.fx.transfer_from(batch_space.output);
}
