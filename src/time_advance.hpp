#pragma once
#include "batch.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this class stores the workspace for time advance -
// a handful of vectors needed to store intermediate RK results
template<typename P>
class explicit_workspace
{
public:
  explicit_workspace(explicit_system<P> const &system)
      : scaled_source(system.x.size()), x_orig(system.x.size()),
        result_1(system.x.size()), result_2(system.x.size()),
        result_3(system.x.size())
  {}

  fk::vector<P> scaled_source;
  fk::vector<P> x_orig;
  fk::vector<P> result_1;
  fk::vector<P> result_2;
  fk::vector<P> result_3;
};

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           explicit_workspace<P> &work,
                           work_set<P> const &batches, P const time,
                           P const dt);

extern template void
explicit_time_advance(PDE<float> const &pde,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      explicit_system<float> &system,
                      explicit_workspace<float> &work,
                      work_set<float> const &batches, float const time,
                      float const dt);
extern template void
explicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      explicit_workspace<double> &work,
                      work_set<double> const &batches, double const time,
                      double const dt);
