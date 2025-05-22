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
  //! sets the precision type
  using precision_type = precision;
  //! allows the creation of a null manager, has to be reinitialized later
  discretization_manager() {
    #ifdef ASGARD_ENABLE_DOUBLE
    #ifdef ASGARD_ENABLE_FLOAT
    static_assert(is_double<precision> or is_float<precision>,
                  "invalid precision type, must use 'double' or 'float'");
    #else
    static_assert(is_double<precision>, "invalid precision type, must use 'double'");
    #endif
    #else
    static_assert(is_float<precision>, "invalid precision type, must use 'float'");
    #endif
  }
  //! take ownership of the pde object and discretize the pde
  discretization_manager(pde_scheme<precision> pde,
                         verbosity_level verbosity = verbosity_level::quiet)
    : verb(pde.options().verbosity.value_or(verbosity)),
      conn(pde.max_level())
  {
    #ifdef ASGARD_ENABLE_DOUBLE
    #ifdef ASGARD_ENABLE_FLOAT
    static_assert(is_double<precision> or is_float<precision>);
    #else
    static_assert(is_double<precision>);
    #endif
    #else
    static_assert(is_float<precision>);
    #endif

    rassert(pde.num_dims() > 0, "cannot discretize an empty pde");

    options_ = std::move(pde.options_);
    domain_  = std::move(pde.domain_);

    initial_md_  = std::move(pde.initial_md_);
    initial_sep_ = std::move(pde.initial_sep_);

    init_compute(); // compute engine, detect GPUs, etc.

    #ifdef ASGARD_USE_MPI
    // only rank 0 will do regular I/O, others will default to silent mode
    if (mpi::comm_rank(options_.mpicomm) != 0)
      verb = verbosity_level::quiet;
    #endif

    if (options_.restarting())
      restart_from_file(pde);
    else
      start_cold(pde);
  }

  //! returns the degree of the discretization
  int degree() const { return hier.degree(); }

  //! returns the number of dimensions
  int num_dims() const { return grid.num_dims(); }
  //! returns the max level of the grid
  int max_level() const { return terms.max_level; }
  //! returns the user provided program options
  prog_opts const &options() const { return options_; }
  //! returns the discretization domain
  pde_domain<precision> const &domain() const { return domain_; }

  //! returns the current simulation time
  double time() const { return stepper.data.time(); }
  //! returns the stop time that, the end of the simulation
  double stop_time() const { return stepper.data.stop_time(); }
  //! returns the time step
  double dt() const { return stepper.data.dt(); }
  //! returns the number of remaining time-steps
  int64_t remaining_steps() const { return stepper.data.num_remain(); }
  //! returns the current time step
  int64_t current_step() const { return stepper.data.step(); }

  //! returns the non-separable initial conditions
  md_func<precision> const &initial_cond_md() const { return initial_md_; }
  //! returns the separable initial conditions
  std::vector<separable_func<precision>> const &initial_cond_sep() const { return initial_sep_; }

  //! set the time in the beginning of the simulation, time() must be zero to call this
  void set_time(precision t) {
    if (stepper.data.step() != 0)
      throw std::runtime_error("cannot reset the current time after the simulation start");
    stepper.data.time() = t;
  }
  //! return the current state, in wavelet format, local to this mpi rank
  std::vector<precision> const &current_state() const { return state; }
  //! returns the size of the current state
  int64_t state_size() const { return static_cast<int64_t>(state.size()); }

  //! return a snapshot of the current solution (in MPI context, only rank 0 gets a valid snapshot)
  reconstruct_solution get_snapshot() const
  {
    #ifdef ASGARD_USE_MPI
    if (not is_leader())
      return reconstruct_solution();
    #endif

    return get_local_snapshot();
  }
  //! return a snapshot of the current solution across all mpi ranks
  reconstruct_solution get_snapshot_mpi() const
  {
    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1)
      terms.resources.bcast(state);
    #endif

    return get_local_snapshot();
  }

  //! check if the terms have poisson dependence
  bool has_poisson() const { return poisson; }
  //! check if the terms have moment dependence
  bool has_moments() const { return moms1d.has_value(); }

  //! computes the right-hand-side of the ode
  void ode_rhs(group_id gid, precision time, std::vector<precision> const &current,
               std::vector<precision> &R) const
  {
    bool constexpr use_groups = true;
    ode_rhs_templ<use_groups>(gid.gid, time, current, R);
  }
  //! computes the right-hand-side of the ode
  void ode_rhs(precision time, std::vector<precision> const &current,
               std::vector<precision> &R) const
  {
    bool constexpr use_groups = false;
    ode_rhs_templ<use_groups>(-1, time, current, R);
  }

  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    bool constexpr use_groups = false;
    ode_rhs_sources<data_mode::replace, use_groups>(-1, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    bool constexpr use_groups = false;
    ode_rhs_sources<data_mode::scal_rep, use_groups>(-1, time, alpha, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    bool constexpr use_groups = false;
    ode_rhs_sources<data_mode::increment, use_groups>(-1, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    bool constexpr use_groups = false;
    ode_rhs_sources<data_mode::scal_inc, use_groups>(-1, time, alpha, src);
  }

  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources_group(group_id gid, precision time, std::vector<precision> &src) const {
    bool constexpr use_groups = true;
    ode_rhs_sources<data_mode::replace, use_groups>(gid.gid, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources_group(group_id gid, precision time, std::vector<precision> &src) const {
    bool constexpr use_groups = true;
    ode_rhs_sources<data_mode::increment, use_groups>(gid.gid, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources_group(group_id gid, precision time, precision alpha, std::vector<precision> &src) const {
    bool constexpr use_groups = true;
    ode_rhs_sources<data_mode::scal_inc, use_groups>(gid.gid, time, alpha, src);
  }

  //! computes the l-2 norm, taking the mass matrix into account
  precision normL2(std::vector<precision> const &x) const {
    expect(x.size() == state.size());
    return terms.normL2(grid, conn, x);
  }

  //! applies all terms
  void terms_apply(precision alpha, std::vector<precision> const &x, precision beta,
                   std::vector<precision> &y) const
  {
    tools::time_event performance_("terms_apply_all kronmult");
    terms.apply_all(grid, conn, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply(precision alpha, precision const x[], precision beta,
                   precision y[]) const
  {
    tools::time_event performance_("terms_apply_all kronmult");
    terms.apply_all(grid, conn, alpha, x, beta, y);
  }
  //! applies terms for the given group
  void terms_apply(group_id gid, precision alpha, std::vector<precision> const &x, precision beta,
                   std::vector<precision> &y) const
  {
    tools::time_event performance_("terms_apply kronmult");
    terms.apply_group(gid.gid, grid, conn, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply(group_id gid, precision alpha, precision const x[], precision beta,
                   precision y[]) const
  {
    tools::time_event performance_("terms_apply kronmult");
    terms.apply_group(gid.gid, grid, conn, alpha, x, beta, y);
  }
  //! applies ADI preconditioner for all terms
  void terms_apply_adi(precision const x[], precision y[]) const
  {
    tools::time_event performance_("terms_apply_adi kronmult");
    terms.apply_all_adi(grid, conn, x, y);
  }

  //! compute the electric field for the given state and update the coefficient matrices
  void do_poisson_update(std::vector<precision> const &field) const;

  //! write out checkpoint/restart data and data for plotting
  void checkpoint() const;
  //! write out snapshot data, same as checkpoint but can be invoked manually
  void save_snapshot(std::filesystem::path const &filename) const;
  //! calls save-snapshot for the final step, if requested with -outfile
  void save_final_snapshot() const
  {
    if (not options_.outfile.empty())
      save_snapshot(options_.outfile);
  }

  //! returns the title of the PDE
  std::string const &title() const { return options_.title; }
  //! returns the subtitle of the PDE
  std::string const &subtitle() const { return options_.subtitle; }
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
    os << "  grid size: " << std::setw(12) << tools::split_style(grid.num_indexes())
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

  //! allows an auxiliary field to be saved for post-processing
  void add_aux_field(aux_field_entry<precision> f) {
    aux_fields.emplace_back(std::move(f));
    if (aux_fields.back().grid.empty()) // if grid provided
      aux_fields.back().grid = grid.get_cells(); // assume the current grid
    rassert(aux_fields.back().data.size()
            == static_cast<size_t>(hier.block_size()
                                   * (aux_fields.back().grid.size() / num_dims())),
            "incompatible data size and number of cells");
  }
  //! return reference to the saved fields
  std::vector<aux_field_entry<precision>> const &
  get_aux_fields() const { return aux_fields; }
  //! deletes the current list of auxiliary fields
  void clear_aux_fields() { aux_fields.clear(); }

#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL
  //! returns a ref to the sparse grid
  sparse_grid const &get_grid() const { return grid; }
  //! returns the current grid generation
  int grid_generation() const { return grid.generation(); }
  //! synchronizes the grid across MPI ranks
  void grid_sync() {
    #ifdef ASGARD_USE_MPI
    grid.mpi_sync(terms.resources, grid_synced_gen_);
    grid_synced_gen_ = grid.generation();
    #endif
  }
  //! returns the term manager
  term_manager<precision> const &get_terms() const { return terms; }
  //! returns the term manager, non-const ref
  term_manager<precision> &get_terms_m() const { return terms; }
  //! returns the compute resources meta structure
  resource_set const &get_resources() const { return terms.resources; }

  //! return the hierarchy_manipulator
  hierarchy_manipulator<precision> const &get_hier() const { return hier; }
  //! return the connection patterns
  connection_patterns const &get_conn() const { return conn; }

  //! recomputes the moments given the state of interest and this term group
  void compute_moments(int groupid, std::vector<precision> const &f) const {
    if ((groupid == -1 and terms.deps().num_moments == 0)
        or (groupid >= 0 and terms.deps(groupid).num_moments == 0)) // no moments needed
      return;

    #ifdef ASGARD_USE_MPI
    if (not is_leader() and not terms.resources.has_moments())
      return;
    #endif

    if (is_leader()) {
      int const level = grid.current_level(0);
      moms1d->project_moments(grid, f, terms.cdata.moments);
      int const num_cells = fm::ipow2(level);
      int const num_outs  = moms1d->num_comp_mom();
      hier.reconstruct1d(
          num_outs, level, span2d<precision>((degree() + 1), num_outs * num_cells,
                                              terms.cdata.moments.data()));
    } else {
      moms1d->resize_moments(grid, terms.cdata.moments);
    }

    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1)
      terms.resources.template bcast
          <precision, resource_comm::moments>(terms.cdata.moments);
    #endif

    if (groupid == -1)
      terms.rebuild_moment_terms(grid, conn, hier);
    else
      terms.rebuild_moment_terms(groupid, grid, conn, hier);
  }
  //! recomputes the moments given the state of interest
  void compute_moments(std::vector<precision> const &f) const {
    compute_moments(-1, f);
  }
  //! recomputes the poisson term for the given group
  void compute_poisson(int groupid, std::vector<precision> const &f) const {
    if (not poisson or (groupid >= 0 and not terms.deps(groupid).poisson))
      return;

    #ifdef ASGARD_USE_MPI
    // leader must always communicate, the rest only if they have a poisson term
    if (not is_leader() and not terms.resources.has_poisson())
      return;
    #endif

    if (is_leader())
      do_poisson_update(f);
    else
      poisson.resize_vector(terms.cdata.electric_field);

    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1)
      terms.resources.template bcast
          <precision, resource_comm::poisson>(terms.cdata.electric_field);
    #endif

    terms.rebuild_poisson(grid, conn, hier);
  }
  //! recomputes the poisson term for the given group
  void compute_poisson(std::vector<precision> const &f) const {
    compute_poisson(-1, f);
  }
  //! (testing/debugging) copy ns to the current state, e.g., force an initial condition
  void set_current_state(std::vector<precision> const &ns) {
    rassert(ns.size() == state.size(), "cannot set state with different size");
    state = ns;
  }
  //! (debugging) prints the term-matrices
  void print_mats() const;

  //! returns true if this is mpi rank 0
  bool is_leader() const { return terms.resources.is_leader(); }

  #ifdef ASGARD_USE_MPI
  //! returns persistent vector for mpi operations
  std::vector<precision> &get_mpiwork() const { return terms.mpiwork; }
  //! sync the state across the mpi communicator
  void sync_mpi_state() const {
    if (terms.resources.num_ranks() > 1)
      terms.resources.bcast(state);
  }
  //! sync the state across the mpi communicator, then return
  std::vector<precision> const &current_state_mpi() const {
    sync_mpi_state();
    return state;
  }
  #else
  void sync_mpi_state() const {}
  std::vector<precision> const &current_state_mpi() const { return state; }
  #endif

  // performs integration in time
  friend void advance_in_time<precision>(
      discretization_manager<precision> &disc, int64_t num_steps);
  // this is the I/O manager
  friend class h5manager<precision>;
  // handles the time-integration meta-data
  friend struct time_advance_manager<precision>;
#endif // __ASGARD_DOXYGEN_SKIP_INTERNAL

protected:
#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL
  //! sets the initial conditions, performs adaptivity in the process
  void set_initial_condition();

  //! start from time 0 and nothing has been set
  void start_cold(pde_scheme<precision> &pde);
  //! restart from a file
  void restart_from_file(pde_scheme<precision> &pde);
  //! common operations for the two start methods
  void start_moments();
  //! computes the right-hand-side of the ode, templated version
  template<bool use_groups>
  void ode_rhs_templ(int gid, precision time, std::vector<precision> const &current,
                     std::vector<precision> &R) const
  {
    if constexpr (use_groups) {
      compute_poisson(gid, current);
      compute_moments(gid, current);
    } else {
      compute_poisson(current);
      compute_moments(current);
    }

    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      terms.mpiwork.resize(current.size());
      // if this rank has terms, then apply_all() will zero out mpiwork/R
      // else an explicit zero-out is needed
      if (is_leader()) {
        terms.resources.bcast(current);
        {
          tools::time_event performance_("ode-rhs kronmult");
          if constexpr (use_groups)
            terms.apply_group(gid, grid, conn, -1, current, 0, terms.mpiwork);
          else
            terms.apply_all(grid, conn, -1, current, 0, terms.mpiwork);
          if (not terms.has_terms()) // mpiwork must be zeroed out explicitly
            std::fill(terms.mpiwork.begin(), terms.mpiwork.end(), 0);
        }{
          tools::time_event performance_("ode-rhs sources");
          if constexpr (use_groups)
            terms.template apply_sources<data_mode::increment>(gid, domain_, grid, conn, hier, time, 1, terms.mpiwork);
          else
            terms.template apply_sources<data_mode::increment>(domain_, grid, conn, hier, time, 1, terms.mpiwork);
        }
        terms.resources.reduce_add(terms.mpiwork, R);
      } else {
        terms.resources.bcast(terms.mpiwork);
        {
          tools::time_event performance_("ode-rhs kronmult");
          if constexpr (use_groups)
            terms.apply_group(gid, grid, conn, -1, terms.mpiwork, 0, R);
          else
            terms.apply_all(grid, conn, -1, terms.mpiwork, 0, R);
          if (not terms.has_terms()) // R must be zeroed out explicitly
            std::fill(R.begin(), R.end(), 0);
        }{
          tools::time_event performance_("ode-rhs sources");
          if constexpr (use_groups)
            terms.template apply_sources<data_mode::increment>(gid, domain_, grid, conn, hier, time, 1, R);
          else
            terms.template apply_sources<data_mode::increment>(domain_, grid, conn, hier, time, 1, R);
        }
        terms.resources.reduce_add(R);
      }
    } else {
    #endif
      {
        tools::time_event performance_("ode-rhs kronmult");
        if constexpr (use_groups)
          terms.apply_group(gid, grid, conn, -1, current, 0, R);
        else
          terms.apply_all(grid, conn, -1, current, 0, R);
        if (not terms.has_terms()) // R wasn't zeroes out above
            std::fill(R.begin(), R.end(), 0);
      }{
        tools::time_event performance_("ode-rhs sources");
        if constexpr (use_groups)
          terms.template apply_sources<data_mode::increment>(gid, domain_, grid, conn, hier, time, 1, R);
        else
          terms.template apply_sources<data_mode::increment>(domain_, grid, conn, hier, time, 1, R);
      }
    #ifdef ASGARD_USE_MPI
    }
    #endif
  }
  //! template version of ode right-hand-side sources
  template<data_mode mode, bool use_groups>
  void ode_rhs_sources(int gid, precision time, precision alpha, std::vector<precision> &src) const {
    tools::time_event performance_("ode sources");
    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      if constexpr (mode == data_mode::replace or mode == data_mode::scal_rep) {
        terms.mpiwork.resize(src.size());
        std::fill(terms.mpiwork.begin(), terms.mpiwork.end(), 0);
      } else {
        terms.mpiwork = src;
      }
      if (is_leader()) {
        if constexpr (use_groups) {
          terms.template apply_sources<mode>(gid, domain_, grid, conn, hier, time, alpha, terms.mpiwork);
        } else {
          terms.template apply_sources<mode>(domain_, grid, conn, hier, time, alpha, terms.mpiwork);
        }
        terms.resources.reduce_add(terms.mpiwork, src);
      } else {
        data_mode constexpr mm = [=]()-> data_mode {
            if constexpr (mode == data_mode::increment)
              return data_mode::replace;
            else if constexpr (mode == data_mode::scal_inc)
              return data_mode::scal_rep;
            else
              return mode;
          }();
        if constexpr (use_groups) {
          terms.template apply_sources<mm>(gid, domain_, grid, conn, hier, time, alpha, src);
        } else {
          terms.template apply_sources<mm>(domain_, grid, conn, hier, time, alpha, src);
        }
        terms.resources.reduce_add(src, src);
      }
    } else {
    #endif
      if constexpr (use_groups) {
        terms.template apply_sources<mode>(gid, domain_, grid, conn, hier, time, alpha, src);
      } else {
        terms.template apply_sources<mode>(domain_, grid, conn, hier, time, alpha, src);
      }
    #ifdef ASGARD_USE_MPI
    }
    #endif
  }
  reconstruct_solution get_local_snapshot() const
  {
    reconstruct_solution shot(
        num_dims(), grid.num_indexes(), grid[0], degree(), state.data());

    std::array<double, max_num_dimensions> xmin, xmax;
    for (int d : iindexof(num_dims())) {
      xmin[d] = domain_.xleft(d);
      xmax[d] = domain_.xright(d);
    }

    shot.set_domain_bounds(xmin.data(), xmax.data());

    return shot;
  }
#endif // __ASGARD_DOXYGEN_SKIP_INTERNAL

private:
  mutable verbosity_level verb = verbosity_level::quiet;
  // user provided options
  prog_opts options_;
  // initial conditions, non-separable
  md_func<precision> initial_md_;
  // initial conditions, separable
  std::vector<separable_func<precision>> initial_sep_;
  // pde-domain
  pde_domain<precision> domain_;

  sparse_grid grid;
  connection_patterns conn;
  hierarchy_manipulator<precision> hier;
  #ifdef ASGARD_USE_MPI
  int grid_synced_gen_ = -2;
  #endif

  // moments
  mutable std::optional<moments1d<precision>> moms1d;
  // poisson solver data
  mutable solvers::poisson<precision> poisson;

  //! term manager holding coefficient matrices and kronmult meta-data
  mutable term_manager<precision> terms;
  //! time advance manager for the different methods
  time_advance_manager<precision> stepper;

  // constantly changing
  #ifdef ASGARD_USE_MPI
  // MPI processes may affect the state, e.g., sync state across the communicator
  mutable
  #endif
  std::vector<precision> state;

  //! fields to store and save for plotting
  std::vector<aux_field_entry<precision>> aux_fields;
};

} // namespace asgard
