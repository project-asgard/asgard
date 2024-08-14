#pragma once

#include "adapt.hpp"
#include "asgard_kron_operators.hpp"
#include "coefficients.hpp"
#include "boundary_conditions.hpp"
#include "moment.hpp"
#include "program_options.hpp"
#include "solver.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "asgard_io.hpp"
#endif

#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

namespace asgard
{

// forward declare so we can declare the fiend time-advance
template<typename precision>
class discretization_manager;

/*!
 * \brief Integrates in time until the final time or number of steps
 *
 * This method manipulates the problems internal state, applying adaptivity,
 * checkpointing and other related operations.
 * The method is decalred as a friend to simplify the implementation is external
 * to simplify the discretization_manager class, which will primarily focus on
 * data storage.
 *
 * The optional variable num_steps indicates the number of time steps to take:
 * - if zero, the method will return immediately,
 * - if negative, integration will continue until the final time step
 */
template<typename P> // implemented in time-advance
void advance_time(discretization_manager<P> &manager, int64_t num_steps = -1);

//! holds matrix and pivot factors
template<typename P>
struct matrix_factor
{
  //! matrix or matrix factors, factorized if ipiv is not empty
  fk::matrix<P> A;
  //! pivots for the factorization
  std::vector<int> ipiv;
};

/*!
 * \brief Wrapper around several aspects of the pde discretization
 *
 * Assumes ownership of the loaded PDE and builds the sparse grid and operators.
 * The current state is set to the initial conditions and time is set to 0
 * (if a restart file is provided the state and time are loaded form the file).
 *
 * Time integration can be performed with the advance_time() function.
 */
template<typename precision>
class discretization_manager
{
public:
  discretization_manager(std::unique_ptr<PDE<precision>> &&pde_in,
                         verbosity_level vebosity = verbosity_level::quiet);

  //! total degrees of freedom for the problem
  int64_t degrees_of_freedom() const
  {
    return grid.size() * fm::ipow(degree_ + 1, pde->num_dims());
  }

  //! returns the degree of the discretization
  int degree() const { return degree_; }

  //! get the current time-step number
  int64_t time_step() const { return time_step_; }

  //! get the current time step
  precision dt() const { return dt_; }
  //! get the current integration time
  precision time() const { return time_; }
  //! get the currently set final time step
  int64_t final_time_step() const { return final_time_step_; }

  //! set new final time step, must be no less than the current time_step()
  void set_final_time_step(int64_t new_final_time_step)
  {
    rassert(new_final_time_step >= time_step_,
            "cannot set the time-step to an easier point in time");
    final_time_step_ = new_final_time_step;
  }
  /*!
   * \brief add new time steps for simulation
   *
   * could add negative number (i.e., subtract time steps) but cannot move
   * the time before the current time_step()
   */
  void add_time_steps(int64_t additional_time_steps)
  {
    rassert(final_time_step_ + additional_time_steps >= time_step_,
            "cannot set the time-step to an easier point in time");
    final_time_step_ += additional_time_steps;
  }

  //! return the current state, in wavelet format, local to this mpi rank
  std::vector<precision> const &current_state() const { return state; }

  //! register the next time step and checkpoint
  void set_next_step(fk::vector<precision> const &next,
                     std::optional<precision> new_dt = {})
  {
    if (new_dt)
      dt_ = new_dt.value();

    state.resize(next.size());
    std::copy(next.begin(), next.end(), state.begin());

    time_ += dt_;

    ++time_step_;

    checkpoint();
  }

  //! write out checkpoint/restart data and data for plotting
  void checkpoint() const;
  //! write out snapshot data, same as checkpoint but can be invoked manually
  void save_snapshot(std::filesystem::path const &filename) const;
  //! calls save-snapshot for the final step, if requested with -outfile
  void save_final_snapshot() const
  {
    if (not pde->options().outfile.empty())
      save_snapshot(pde->options().outfile);
  }

  //! if analytic solution exists, return the rmse error
  std::optional<std::array<std::vector<precision>, 2>> rmse_exact_sol() const;

  //! collect the current state from across all mpi ranks
  fk::vector<precision> current_mpistate() const;

  //! returns a ref to the original pde
  PDE<precision> const &get_pde() const { return *pde; }
  //! returns a ref to the sparse grid
  adapt::distributed_grid<precision> const &get_grid() const { return grid; }
  //! return the transformer
  auto const &get_transformer() const { return transformer; }
  //! return the fixed boundary conditions
  auto const &get_fixed_bc() const { return fixed_bc; }

  //! get kronopts, return the kronmult operators for iterative solvers
  kron_operators<precision> &get_kronops() const { return kronops; }
  //! return operator matrix for direct solves
  std::optional<matrix_factor<precision>> &get_op_matrix() const { return op_matrix; }
  //! returns the moments
  std::vector<moment<precision>> &get_moments() const { return moments; }

  friend void advance_time<precision>(discretization_manager<precision> &, int64_t);

protected:
  //! convenient check if we are using high verbosity level
  bool high_verbosity() const { return (verb == verbosity_level::high); }
  //! update components on grid reset
  void update_grid_components()
  {
    kronops.clear();
    auto const my_subgrid = grid.get_subgrid(get_rank());
    fixed_bc = boundary_conditions::make_unscaled_bc_parts(
        *pde, grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);
    if (op_matrix)
      op_matrix.reset();
    if (not moments.empty()
        and pde->options().step_method.value() == time_advance::method::imex)
      reset_moments();
  }
  void reset_moments()
  {
    tools::time_event performance("update_system");

    int const level      = pde->get_dimensions()[0].get_level();
    precision const min  = pde->get_dimensions()[0].domain_min;
    precision const max  = pde->get_dimensions()[0].domain_max;
    int const N_elements = fm::two_raised_to(level);

    int const quad_dense_size = dense_dim_size(ASGARD_NUM_QUADRATURE - 1, level);

    for (auto &m : moments)
    {
      m.createFlist(*pde);
      expect(m.get_fList().size() > 0);

      m.createMomentVector(*pde, grid.get_table());
      expect(m.get_vector().size() > 0);

      m.createMomentReducedMatrix(*pde, grid.get_table());
    }

    if (pde->do_poisson_solve())
    {
      // Setup poisson matrix initially
      solver::setup_poisson(N_elements, min, max, pde->poisson_diag,
                            pde->poisson_off_diag);
    }

    pde->E_field.resize(quad_dense_size);
    pde->phi.resize(quad_dense_size);
    pde->E_source.resize(quad_dense_size);
  }

private:
  verbosity_level verb;
  std::unique_ptr<PDE<precision>> pde;

  adapt::distributed_grid<precision> grid;

  basis::wavelet_transform<precision, resource::host> transformer;

  // easy access variables, avoids jumping into pde->options()
  int degree_;

  // extra parameters
  precision dt_;
  precision time_;
  int64_t time_step_;

  int64_t final_time_step_;

  // recompute only when the grid changes
  // left-right boundary conditions, time-independent components
  std::array<boundary_conditions::unscaled_bc_parts<precision>, 2> fixed_bc;

  // used for iterative solvers
  mutable kron_operators<precision> kronops;
  // used for direct solvers
  mutable std::optional<matrix_factor<precision>> op_matrix;
  // moments of the field
  mutable std::vector<moment<precision>> moments;

  // constantly changing
  std::vector<precision> state;
};

} // namespace asgard
