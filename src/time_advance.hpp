#pragma once
#include "adapt.hpp"
#include "asgard_kronmult_matrix.hpp"
#include "boundary_conditions.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

namespace asgard::time_advance
{
enum class method
{
  imp,
  exp, // explicit is reserved keyword
  imex
};

#ifdef ASGARD_USE_CUDA
static constexpr resource imex_resrc = resource::device;
#else
static constexpr resource imex_resrc = resource::host;
#endif

// take an adaptivity-enabled timestep
// make require many "pseudosteps" to refine
template<typename P>
fk::vector<P>
adaptive_advance(method const step_method, PDE<P> &pde,
                 matrix_list<P> &operator_matrices,
                 adapt::distributed_grid<P> &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts, fk::vector<P> const &x,
                 P const time, bool const update_system = false);

// this function executes a time step using the current solution
// vector x (in host_space).
// on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_advance(PDE<P> const &pde, matrix_list<P> &operator_matrices,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                     &unscaled_parts,
                 fk::vector<P> const &x, P const time);

template<typename P>
fk::vector<P>
implicit_advance(PDE<P> &pde, matrix_list<P> &operator_matrices,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                     &unscaled_parts,
                 fk::vector<P> const &x, P const time,
                 bool const update_system = true);

template<typename P>
fk::vector<P>
imex_advance(PDE<P> &pde, matrix_list<P> &operator_matrices,
             adapt::distributed_grid<P> const &adaptive_grid,
             basis::wavelet_transform<P, resource::host> const &transformer,
             options const &program_opts,
             std::array<boundary_conditions::unscaled_bc_parts<P>, 2> const
                 &unscaled_parts,
             fk::vector<P> const &x_orig, fk::vector<P> const &x_prev,
             P const time, solve_opts const solver,
             bool const update_system = true);

} // namespace asgard::time_advance
