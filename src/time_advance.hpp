#pragma once
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "kronmult.hpp"
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
                      fk::vector<P> const &x, distribution_plan const &plan,
                      int const workspace_size_MB, P const time);

template<typename P>
fk::vector<P>
implicit_time_advance(PDE<P> const &pde, element_table const &table,
                      std::vector<fk::vector<P>> const &unscaled_sources,
                      std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                      fk::vector<P> const &x,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, P const time,
                      solve_opts const solver  = solve_opts::direct,
                      bool const update_system = true);
