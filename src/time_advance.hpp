#pragma once
#include "batch.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x (in host_space).
// on exit, the next solution vector is stored in x.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           device_workspace<P> &dev_space,
                           std::vector<element_chunk> const &chunks,
                           distribution_plan const &plan, P const time,
                           P const dt);

template<typename P>
void implicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           std::vector<element_chunk> const &chunks,
                           P const time, P const dt, bool update_system = true);
