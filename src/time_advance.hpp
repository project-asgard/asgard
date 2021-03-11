#pragma once
#include "adapt.hpp"
#include "boundary_conditions.hpp"
#include "kronmult.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

namespace time_advance
{
enum class method
{
  imp,
  exp // explicit is reserved keyword
};

// take an adaptivity-enabled timestep
// make require many "pseudosteps" to refine
template<typename P>
fk::vector<P>
adaptive_advance(method const step_method, PDE<float> &sp_pde, PDE<P> &pde,
                 adapt::distributed_grid<P> &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts, fk::vector<P> const &x,
                 P const time, int const workspace_size_MB,
                 bool const update_system = false);

// this function executes a time step using the current solution
// vector x (in host_space).
// on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_advance(PDE<float> const &sp_pde, PDE<P> const &pde,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                 fk::vector<P> const &x, int const workspace_size_MB,
                 P const time);

template<typename P>
fk::vector<P>
implicit_advance(PDE<P> const &pde,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                 fk::vector<P> const &x, P const time,
                 solve_opts const solver  = solve_opts::direct,
                 bool const update_system = true);

} // namespace time_advance
