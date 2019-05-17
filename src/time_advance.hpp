#pragma once
#include "batch.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, fk::vector<P> const &x,
                           fk::vector<P> const &fx,
                           std::vector<batch_operands_set<P>> const &batches,
                           std::vector<fk::vector<P>> &unscaled_sources,
                           fk::vector<P> &scaled_source, P const time);

extern template void
explicit_time_advance(PDE<float> const &pde, fk::vector<float> const &x,
                      fk::vector<float> const &fx,
                      std::vector<batch_operands_set<float>> const &batches,
                      std::vector<fk::vector<float>> &unscaled_sources,
                      fk::vector<float> &scaled_source, float const time);

extern template void
explicit_time_advance(PDE<double> const &pde, fk::vector<double> const &x,
                      fk::vector<double> const &fx,
                      std::vector<batch_operands_set<double>> const &batches,
                      std::vector<fk::vector<double>> &unscaled_sources,
                      fk::vector<double> &scaled_source, double const time);
