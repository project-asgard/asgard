#pragma once
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "timer.hpp"
#include <mutex>

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
                      solve_opts const solver  = solve_opts::direct,
                      bool const update_system = true);

// apply the system matrix to the current solution vector using batched
// gemm.
static std::once_flag print_flag;
template<typename P>
static fk::vector<P>
apply_A(PDE<P> const &pde, element_table const &elem_table,
        element_subgrid const &grid, std::vector<element_chunk> const &chunks,
        fk::vector<P> const &x)
{
  batch_workspace<P, resource::device> batch_space(pde, grid, chunks);

  // print information about workspace size on first invocation
  std::call_once(print_flag, [&batch_space] {
    auto const get_MB = [&](int num_elems) {
      int64_t const bytes    = num_elems * sizeof(P);
      double const megabytes = bytes * 1e-6;
      return megabytes;
    };

    node_out() << "batch workspace size..." << '\n';

    node_out() << "input vector size (MB): " << get_MB(batch_space.input.size())
               << '\n';
    node_out() << "kronmult output space size (MB): "
               << get_MB(batch_space.reduction_space.size()) << '\n';
    node_out() << "kronmult working space size (MB): "
               << get_MB(batch_space.kron_intermediate.size()) << '\n';
    node_out() << "output vector size (MB): "
               << get_MB(batch_space.output.size()) << '\n';
    auto const &unit_vect = batch_space.get_unit_vector();
    node_out() << "reduction vector size (MB): " << get_MB(unit_vect.size())
               << '\n';
  });

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
  return batch_space.output.clone_onto_host();
}
