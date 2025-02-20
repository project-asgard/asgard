#pragma once
#include "asgard_time_advance.hpp"

#ifdef ASGARD_USE_HIGHFIVE
#include "asgard_io.hpp"
#endif

/*!
 * \internal
 * \file asgard_discretization.hpp
 * \brief Defines the container class discretization_manager
 * \author The ASGarD Team
 * \ingroup asgard_discretization
 *
 * \endinternal
 */

namespace asgard
{

/*!
 * \ingroup asgard_discretization
 * \brief Wrapper around several aspects of the pde discretization
 *
 * Assumes ownership of the loaded PDE and builds the sparse grid and operators.
 * The current state is set to the initial conditions and time is set to 0
 * (if a restart file is provided the state and time are loaded form the file).
 *
 * Time integration can be performed with the advance_time() function.
 */
template<typename precision = default_precision>
class discretization_manager
{
public:
  //! take ownership of the pde object and discretize the pde
  discretization_manager(std::unique_ptr<PDE<precision>> &&pde_in,
                         verbosity_level verbosity = verbosity_level::quiet);

  //! take ownership of the pde object and discretize the pde
  discretization_manager(PDEv2<precision> pde_in,
                         verbosity_level verbosity = verbosity_level::quiet);

  /*!
   * \brief Preventing relocation
   *
   * Different components of the manager can hold aliases (pointer and refs)
   * to other components, e.g., components shared between multiple other components
   * such as scratch workspaces or common pde options.
   * Relocating the manager can break all of those references, thus we explicitly
   * forbid such operations.
   * If "move" operations are needed, wrap the manger in a unique_ptr.
   */
  discretization_manager(discretization_manager &&) = delete;

  //! total degrees of freedom for the problem
  int64_t degrees_of_freedom() const
  {
    return grid.size() * fm::ipow(degree_ + 1, pde->num_dims());
  }

  //! returns the degree of the discretization
  int degree() const { return degree_; }

  //! returns the number of dimensions
  int num_dims() const { return pde2.num_dims(); }

  //! returns the time discretization parameters
  time_data<precision> const &time_params() const { return stepper.data; }

  //! get the current time-step number
  int64_t time_step() const { return time_step_; }

  //! get the current time-step size
  precision dt() const { return dt_; }
  //! get the current integration time
  precision time() const { return time_; }
  //! set the time in the befinning of the simulation, time() must be zero to call this
  void set_time(precision t) {
    if (stepper.data.step() != 0)
      throw std::runtime_error("cannot reset the current time after the simulation start");
    stepper.data.time() = t;
  }
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
  //! returns the size of the current state
  int64_t state_size() const { return static_cast<int64_t>(state.size()); }

  //! return a snapshot of the current solution
  reconstruct_solution get_snapshot() const
  {
    if (pde) { // version 1
      reconstruct_solution shot(
          pde->num_dims(), grid.size(), grid.get_table().get_active_table().data(),
          degree_, state.data());

      std::array<double, max_num_dimensions> xmin, xmax;
      for (int d : iindexof(pde->num_dims())) {
        xmin[d] = pde->get_dimensions()[d].domain_min;
        xmax[d] = pde->get_dimensions()[d].domain_max;
      }

      shot.set_domain_bounds(xmin.data(), xmax.data());

      return shot;
    } else {
      reconstruct_solution shot(
          pde2.num_dims(), sgrid.num_indexes(), sgrid[0], degree_, state.data(), true);

      std::array<double, max_num_dimensions> xmin, xmax;
      for (int d : iindexof(pde2.num_dims())) {
        xmin[d] = pde2.domain().xleft(d);
        xmax[d] = pde2.domain().xright(d);
      }

      shot.set_domain_bounds(xmin.data(), xmax.data());

      return shot;
    }
  }

  //! computes the right-hand-side of the ode
  void ode_rhs(imex_flag imflag, precision time, std::vector<precision> const &state,
               std::vector<precision> &R) const;
  //! computes the right-hand-side of the backward Euler method
  void ode_irhs(precision time, std::vector<precision> const &state,
                std::vector<precision> &R) const;
  //! solves x = A^{-1} x where A is the kron_operators with given flag, uses method from options
  void ode_sv(imex_flag imflag, std::vector<precision> &x) const;

  //! computes the right-hand-side of the ode
  void ode_rhs_v2(precision time, std::vector<precision> const &current,
                  std::vector<precision> &R) const
  {
    if (poisson) { // if we have a Poisson dependence
      tools::time_event performance_("ode-rhs poisson");
      do_poisson_update(current);
      terms.rebuild_poisson(sgrid, conn, hier);
    }

    {
      tools::time_event performance_("ode-rhs kronmult");
      terms.apply_all(sgrid, conn, -1, current, 0, R);
    }{
      tools::time_event performance_("ode-rhs sources");
      terms.template apply_sources<data_mode::increment>(pde2.domain(), sgrid, conn, hier, time, 1, R);
    }
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    tools::time_event performance_("set ode sources");
    terms.template apply_sources<data_mode::replace>(pde2.domain(), sgrid, conn, hier, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    tools::time_event performance_("set ode sources");
    terms.template apply_sources<data_mode::scal_rep>(pde2.domain(), sgrid, conn, hier, time, alpha, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    tools::time_event performance_("set ode sources");
    terms.template apply_sources<data_mode::increment>(pde2.domain(), sgrid, conn, hier, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    tools::time_event performance_("set ode sources");
    terms.template apply_sources<data_mode::scal_inc>(pde2.domain(), sgrid, conn, hier, time, alpha, src);
  }

  //! applies all terms
  void terms_apply_all(precision alpha, std::vector<precision> const &x, precision beta,
                       std::vector<precision> &y) const
  {
    tools::time_event performance_("terms_apply_all kronmult");
    terms.apply_all(sgrid, conn, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply_all(precision alpha, precision const x[], precision beta,
                       precision y[]) const
  {
    tools::time_event performance_("terms_apply_all kronmult");
    terms.apply_all(sgrid, conn, alpha, x, beta, y);
  }
  //! applies ADI preconditioner for all terms
  void terms_apply_adi(precision const x[], precision y[]) const
  {
    tools::time_event performance_("terms_apply_adi kronmult");
    terms.apply_all_adi(sgrid, conn, x, y);
  }

  //! compute the electric field for the given state and update the coefficient matrices
  void do_poisson_update(std::vector<precision> const &field) const;

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
    if (pde) {
      if (not pde->options().outfile.empty())
        save_snapshot(pde->options().outfile);
    } else {
      if (not pde2.options().outfile.empty())
        save_snapshot(pde2.options().outfile);
    }
  }

  /*!
   * \brief if analytic solution exists, return the rmse error
   *
   * If no analytic solution has been specified, the optional will be empty.
   * If an analytic solution exists, this will return both the absolute and
   * relative rmse (normalized by the max entry of the exact solution).
   * The vector contains an entry for each mpi rank.
   *
   * (note: we are working on computing the rmse for all mpi ranks instead
   * of per rank)
   */
  std::optional<std::array<std::vector<precision>, 2>> rmse_exact_sol() const;
  /*!
   * \brief returns the vector of the exact solution at the current time step
   *
   * If no analytic solution has been specified, the optional will be empty.
   * If the solution has been specified, this will return the exact solution
   * at the current time and projected on the current grid.
   */
  std::optional<std::vector<precision>> get_exact_solution() const;

  //! collect the current state from across all mpi ranks
  fk::vector<precision> current_mpistate() const;

  //! returns a ref to the original pde
  PDE<precision> const &get_pde() const { return *pde; }
  //! returns a ref to the sparse grid
  adapt::distributed_grid<precision> const &get_grid() const { return grid; }

  //! returns the title of the PDE
  std::string const &title() const { return pde2.options().title; }
  //! returns the subtitle of the PDE
  std::string const &subtitle() const { return pde2.options().subtitle; }
  //! returns true if the title contains the given sub-string
  bool title_contains(std::string const &substring) const {
    return (title().find(substring) != std::string::npos);
  }
  //! returns true if the subtitle contains the given sub-string
  bool subtitle_contains(std::string const &substring) const {
    return (subtitle().find(substring) != std::string::npos);
  }

  //! convenient check if we are using high verbosity level
  bool high_verbosity() const { return (verb == verbosity_level::high); }
  //! convenient check if we are using low verbosity level
  bool low_verbosity() const { return (verb == verbosity_level::low); }
  //! convenient check if we are using quiet verbosity level
  bool stop_verbosity() const { return (verb == verbosity_level::quiet); }
  //! resets the verbosity level
  void set_verbosity(verbosity_level v) const { verb = v; }

  //! integrate in time for the given number of steps, -1 means until the end
  void advance_time(int64_t num_steps = -1) {
    advance_in_time(*this, num_steps);
  }

  //! report time progress
  void progress_report(std::ostream &os = std::cout) const {
    if (stepper.is_steady_state())
    {
      os << "refinement iteration " << std::setw(10) << tools::split_style(stepper.data.step());
    }
    else
    {
      os << "time-step: " << std::setw(10) << tools::split_style(stepper.data.step()) << "  time: ";
      std::string s = std::to_string(stepper.data.time());

      if (s.size() < 7)
        os << std::setw(10) << s << std::string(7 - s.size(), ' ');
      else
        os << std::setw(10) << s;
    }
    os << "  grid size: " << std::setw(12) << tools::split_style(sgrid.num_indexes())
       << "  dof: " << std::setw(14) << tools::split_style(state.size());
    int64_t const num_appy = stepper.solver_iterations();
    if (num_appy > 0) { // using iterative solver
      os << "  av-iter: " << std::setw(14) << tools::split_style(num_appy / stepper.data.step())
         << '\n';
    } else {
      os << '\n';
    }
  }
  //! safe final result and print statistics, if verbosity allows it and output file is given
  void final_output() const {
    save_final_snapshot();
    if (not stop_verbosity()) {
      progress_report();
      if (asgard::tools::timer.enabled())
        std::cout << asgard::tools::timer.report() << '\n';
    }
  }

  //! projects and sum-of-separable functions and md_func onto the current basis
  void project_function(std::vector<separable_func<precision>> const &sep,
                        md_func<precision> const &fmd, std::vector<precision> &out) const;

  //! projects and sum-of-separable functions and md_func onto the current basis
  std::vector<precision> project_function(
      std::vector<separable_func<precision>> const &sep = {},
      md_func<precision> const &fmd = nullptr) const
  {
    if (sep.empty() and not fmd)
      return std::vector<precision>(state.size());

    std::vector<precision> result;
    project_function(sep, fmd, result);
    return result;
  }
  //! projects a single separable function and md_func onto the current basis
  std::vector<precision> project_function(
      separable_func<precision> const &sep = {},
      md_func<precision> const &fmd = nullptr) const
  {
    std::vector<precision> result;
    project_function({sep, }, fmd, result);
    return result;
  }

#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL

  PDEv2<precision> const &get_pde2() const { return pde2; }
  time_data<precision> const &time_props() const { return stepper.data; }
  bool version2() const { return not pde; }
  void save_snapshot2(std::filesystem::path const &filename) const;
  sparse_grid const &get_sgrid() const { return sgrid; }

  term_manager<precision> const & get_terms() const { return terms; }

  //! return the hierarchy_manipulator
  auto const &get_hiermanip() const { return hier; }
  //! return the fixed boundary conditions
  auto const &get_fixed_bc() const { return fixed_bc; }
  //! return the connection patterns
  auto const &get_conn() const { return conn; }

  //! get kronopts, return the kronmult operators for iterative solvers
  kron_operators<precision> &get_kronops() const { return kronops; }
  //! return operator matrix for direct solves
  std::optional<matrix_factor<precision>> &get_op_matrix() const { return op_matrix; }
  //! returns the coefficient matrices
  coefficient_matrices<precision> &get_cmatrices() const { return matrices; }
  //! recomputes the moments given the state of interest
  void compute_moments(std::vector<precision> const &f) {
    if (moms1d) {
      int const level = pde->get_dimensions().front().get_level();
      moms1d->project_moments(level, f, grid.get_table(), matrices.edata.moments);
      int const num_cells = fm::ipow2(level);
      int const num_outs  = moms1d->num_comp_mom();
      hier.reconstruct1d(
          num_outs, level, span2d<precision>((degree_ + 1), num_outs * num_cells,
                                             matrices.edata.moments.data()));
    }
  }
  //! (testing) recomputes the moments given the state of interest, keeps in hierarchical form
  void compute_hmoments(std::vector<precision> const &f, std::vector<precision> &rmom) {
    if (moms1d)
      moms1d->project_moments(pde->get_dimensions().front().get_level(),
                              f, grid.get_table(), rmom);
  }
  void set_current_state(std::vector<precision> const &ns) {
    rassert(ns.size() == state.size(), "cannot set state with different size");
    state = ns;
  }
  //! recomputes the coefficients, can select sub
  void compute_coefficients(coeff_update_mode mode = coeff_update_mode::all) {
    generate_coefficients(*pde, matrices, conn, hier, time_, mode);
#ifndef KRON_MODE_GLOBAL
    pde->coeffs_.resize(pde->num_terms() * pde->num_dims());
    for (int64_t t : indexof(pde->coeffs_.size()))
      pde->coeffs_[t] = matrices.term_coeffs[t].to_fk_matrix(degree_ + 1, conn);
#endif
  }
  fk::matrix<precision> get_coeff_matrix(int t, int d) const
  {
    return matrices.term_coeffs[t * pde->num_dims() + d].to_fk_matrix(hier.degree() + 1, conn);
  }
  /*!
   * \ingroup asgard_discretization
   * \brief Performs integration in time for a given number of steps
   */
  friend void advance_in_time<precision>(discretization_manager<precision> &disc,
                                      int64_t num_steps);

  friend void advance_time_v2<precision>(discretization_manager<precision> &disc,
                                         int64_t num_steps);

  friend class h5manager<precision>;

  friend struct time_advance_manager<precision>;
#endif // __ASGARD_DOXYGEN_SKIP_INTERNAL

protected:
#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL
  //! sets the initial conditions, performs adaptivity in the process
  void set_initial_condition_v1();
  //! sets the initial conditions, performs adaptivity in the process
  void set_initial_condition();
  //! update components on grid reset
  void update_grid_components()
  {
    tools::time_event performance("update grid components");
    kronops.clear();
    generate_coefficients(*pde, matrices, conn, hier, time_, coeff_update_mode::independent);

#ifdef KRON_MODE_GLOBAL
    // the imex-flag is not used internally
    kronops.make(imex_flag::unspecified, *pde, matrices, grid);
#else
    pde->coeffs_.resize(pde->num_terms() * pde->num_dims());
    for (int64_t t : indexof(pde->coeffs_.size()))
      pde->coeffs_[t] = matrices.term_coeffs[t].to_fk_matrix(degree_ + 1, conn);
#endif

    auto const my_subgrid = grid.get_subgrid(get_rank());
    fixed_bc = boundary_conditions::make_unscaled_bc_parts(
        *pde, grid.get_table(), hier, matrices,
        conn, my_subgrid.row_start, my_subgrid.row_stop);
    if (op_matrix)
      op_matrix.reset();
  }

  //! start from time 0 and nothing has been set
  void start_cold();
  //! restart from a file
  void restart_from_file();

#endif // __ASGARD_DOXYGEN_SKIP_INTERNAL

private:
  mutable verbosity_level verb;
  std::unique_ptr<PDE<precision>> pde;
  PDEv2<precision> pde2;

  adapt::distributed_grid<precision> grid;

  sparse_grid sgrid;

  connection_patterns conn;

  hierarchy_manipulator<precision> hier; // new transformer

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

  // stores the coefficient matrices
  mutable coefficient_matrices<precision> matrices;
  // used for all separable operations and iterative solvers
  mutable kron_operators<precision> kronops;
  // used for direct solvers
  mutable std::optional<matrix_factor<precision>> op_matrix;
  // moments, new implementation
  mutable std::optional<moments1d<precision>> moms1d;
  // poisson solver data
  mutable solvers::poisson<precision> poisson;

  //! term manager holding coefficient matrices and kronmult meta-data
  mutable term_manager<precision> terms;
  //! source manager holding the sources and inhomogeneous boundary conditions
  // mutable source_manager<precision> sources;
  //! time advance manager for the different methods
  time_advance_manager<precision> stepper;

  // constantly changing
  std::vector<precision> state;
};

} // namespace asgard
