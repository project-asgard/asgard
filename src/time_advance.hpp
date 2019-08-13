#pragma once
#include "batch.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           work_set<P> const &batches, P const time,
                           P const dt);

template<typename P>
void implicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           work_set<P> const &batches, P const time, P const dt,
                           fk::matrix<P> &A);
template<typename P>
void implicit_iterative_time_advance(PDE<P> const &pde,
    std::vector<fk::vector<P>> const &unscaled_sources,
    explicit_system<P> &system, work_set<P> const &batches, P const time,
    P const dt, fk::matrix<P> &A);

extern template void
explicit_time_advance(PDE<float> const &pde,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      explicit_system<float> &system,
                      work_set<float> const &batches, float const time,
                      float const dt);
extern template void
explicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      work_set<double> const &batches, double const time,
                      double const dt);

extern template void
implicit_time_advance(PDE<float> const &pde,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      explicit_system<float> &system,
                      work_set<float> const &batches, float const time,
                      float const dt, fk::matrix<float> &A);
extern template void
implicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      work_set<double> const &batches, double const time,
                      double const dt, fk::matrix<double> &A);

//extern template void
//implicit_iterative_time_advance(PDE<float> const &pde,
//                      std::vector<fk::vector<float>> const &unscaled_sources,
//                      explicit_system<float> &system,
//                      work_set<float> const &batches, float const time,
//                      float const dt, fk::matrix<float> &A);
extern template void
implicit_iterative_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      work_set<double> const &batches, double const time,
                      double const dt, fk::matrix<double> &A);