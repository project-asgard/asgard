#pragma once
#include "batch.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, fk::vector<P> &x,
                           fk::vector<P> &x_orig, fk::vector<P> &fx,
                           fk::vector<P> &scaled_source,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           std::vector<fk::vector<P>> &workspace,

                           std::vector<batch_operands_set<P>> const &batches,
                           P const time, P const dt);

extern template void
explicit_time_advance(PDE<float> const &pde, fk::vector<float> &x,
                      fk::vector<float> &x_orig, fk::vector<float> &fx,
                      fk::vector<float> &scaled_source,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      std::vector<fk::vector<float>> &workspace,

                      std::vector<batch_operands_set<float>> const &batches,
                      float const time, float const dt);

extern template void
explicit_time_advance(PDE<double> const &pde, fk::vector<double> &x,
                      fk::vector<double> &x_orig, fk::vector<double> &fx,
                      fk::vector<double> &scaled_source,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      std::vector<fk::vector<double>> &workspace,
                      std::vector<batch_operands_set<double>> const &batches,
                      double const time, double const dt);
