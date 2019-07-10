#pragma once
#include "batch.hpp"
#include "grouping.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           rank_workspace<P> &rank_space,
                           std::vector<element_group> groups, P const time,
                           P const dt);

extern template void
explicit_time_advance(PDE<float> const &pde, element_table const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_group> groups, float const time,
                      float const dt);

extern template void
explicit_time_advance(PDE<double> const &pde, element_table const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double> &host_space,
                      rank_workspace<double> &rank_space,
                      std::vector<element_group> groups, double const time,
                      double const dt);
