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
    : verb(pde.options().verbosity.value_or(verbosity))
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
  //! extend the number of steps
  void extend_steps(int64_t num_more) { stepper.data.extend_steps(num_more); }

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
  bool has_moments() const { return !!terms.moms; }

  //! computes the right-hand-side of the ode
  void ode_rhs(group_id gid, precision time, std::vector<precision> const &current,
               std::vector<precision> &R) const
  {
    ode_rhs_base(gid.gid, time, current, R);
  }
  //! computes the right-hand-side of the ode
  void ode_rhs(precision time, std::vector<precision> const &current,
               std::vector<precision> &R) const
  {
    ode_rhs_base(all_groups, time, current, R);
  }
  //! takes an Euler-like step, next = current + ode-rhs(current), but terms and sources can be scaled separately
  void ode_euler(group_id gid, precision time, std::vector<precision> const &current,
                 terms_scale term_scal, sources_scale source_scal,
                 std::vector<precision> &next) const
  {
    ode_euler_base(gid.gid, time, current, term_scal, source_scal, next);
  }
  //! takes an Euler-like step, next = current + scale * ode-rhs(current)
  void ode_euler(group_id gid, precision time, std::vector<precision> const &current,
                 precision scale, std::vector<precision> &next) const
  {
    ode_euler_base(gid.gid, time, current, terms_scale{scale}, sources_scale{scale}, next);
  }
  //! takes an Euler-like step, next = current + ode-rhs(current), but terms and sources can be scaled separately
  void ode_euler(precision time, std::vector<precision> const &current,
                 terms_scale term_scal, sources_scale source_scal,
                 std::vector<precision> &next) const
  {
    ode_euler_base(all_groups, time, current, term_scal, source_scal, next);
  }
  //! takes an Euler-like step, next = current + scale * ode-rhs(current)
  void ode_euler(precision time, std::vector<precision> const &current,
                 precision scale, std::vector<precision> &next) const
  {
    ode_euler_base(all_groups, time, current, terms_scale{scale}, sources_scale{scale}, next);
  }

  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::replace>(all_groups, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::scal_rep>(all_groups, time, alpha, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::increment>(all_groups, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources(precision time, precision alpha, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::scal_inc>(all_groups, time, alpha, src);
  }

  //! computes the ode right-hand-side sources by projecting them onto the basis and setting them in src
  void set_ode_rhs_sources_group(group_id gid, precision time, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::replace>(gid.gid, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources_group(group_id gid, precision time, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::increment>(gid.gid, time, 1, src);
  }
  //! computes the ode right-hand-side sources by projecting them onto the basis and adding them to src
  void add_ode_rhs_sources_group(group_id gid, precision time, precision alpha, std::vector<precision> &src) const {
    ode_rhs_sources<data_mode::scal_inc>(gid.gid, time, alpha, src);
  }

  //! computes the l-2 norm, taking the mass matrix into account
  precision normL2(std::vector<precision> const &x) const {
    expect(x.size() == state.size());
    return terms.normL2(grid, conn, x);
  }

  //! applies all terms, does not recompute moments
  void terms_apply(precision alpha, std::vector<precision> const &x, precision beta,
                   std::vector<precision> &y) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(-1, grid, conn);
    tools::time_event performance_("terms_apply_all kronmult", flops);
    #else
    tools::time_event performance_("terms_apply_all kronmult");
    #endif
    terms.apply(grid, conn, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply(precision alpha, precision const x[], precision beta,
                   precision y[]) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(-1, grid, conn);
    tools::time_event performance_("terms_apply_all kronmult", flops);
    #else
    tools::time_event performance_("terms_apply_all kronmult");
    #endif
    terms.apply(grid, conn, alpha, x, beta, y);
  }
  //! applies terms for the given group, does not recompute moments
  void terms_apply(group_id gid, precision alpha, std::vector<precision> const &x, precision beta,
                   std::vector<precision> &y) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(gid.gid, grid, conn);
    tools::time_event performance_("terms_apply kronmult", flops);
    #else
    tools::time_event performance_("terms_apply kronmult");
    #endif
    terms.apply(gid.gid, grid, conn, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply(group_id gid, precision alpha, precision const x[], precision beta,
                   precision y[]) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(gid.gid, grid, conn);
    tools::time_event performance_("terms_apply kronmult", flops);
    #else
    tools::time_event performance_("terms_apply kronmult");
    #endif
    terms.apply(gid.gid, grid, conn, alpha, x, beta, y);
  }
  #ifdef ASGARD_USE_GPU
  //! applies all terms, non-owning array signature
  void terms_apply_gpu(precision alpha, precision const x[], precision beta,
                       precision y[]) const
  {
    terms_apply_gpu(group_id{term_manager<precision>::all_groups}, alpha, x, beta, y);
  }
  //! applies all terms, non-owning array signature
  void terms_apply_gpu(group_id gid, precision alpha, precision const x[], precision beta,
                       precision y[]) const
  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(gid.gid, grid, conn);
    tools::time_event performance_("terms_apply kronmult", flops);
    #else
    tools::time_event performance_("terms_apply kronmult");
    #endif
    terms.apply_gpu(gid.gid, grid, conn, alpha, x, beta, y);
  }
  #endif

  #ifdef ASGARD_USE_MPI
  //! initiate iterative loop on MPI for the given group and workspace
  void mpi_iteration_apply(group_id gid, std::vector<precision> &work) const {
    mpi_iteration_apply_base(gid.gid, work);
  }
  //! initiate iterative loop on MPI for the all groups and given workspace
  void mpi_iteration_apply(std::vector<precision> &work) const {
    mpi_iteration_apply_base(all_groups, work);
  }
  //! stop the currently working iteration
  void mpi_iteration_stop() const;
  //! performs apply operation on the leader, assuming the non-leader ranks are running mpi_iteration_apply()
  void mpi_leader_apply(precision alpha, precision const x[], precision beta,
                        precision y[]) const
  {
    mpi_leader_apply_base(all_groups, alpha, x, beta, y);
  }
  //! performs apply operation on the leader, assuming the non-leader ranks are running mpi_iteration_apply()
  void mpi_leader_apply(group_id gid, precision alpha, precision const x[],
                        precision beta, precision y[]) const
  {
    mpi_leader_apply_base(gid.gid, alpha, x, beta, y);
  }
  #else
  void mpi_iteration_apply(group_id, std::vector<precision> &) const {}
  void mpi_iteration_apply(std::vector<precision> &) const {}
  void mpi_iteration_stop() const {}
  void mpi_leader_apply(precision alpha, precision const x[], precision beta,
                        precision y[]) const
  {
    terms_apply(alpha, x, beta, y);
  }
  void mpi_leader_apply(group_id gid, precision alpha, precision const x[],
                        precision beta, precision y[]) const
  {
    terms_apply(gid, alpha, x, beta, y);
  }
  #endif

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

    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = tools::timer.max_flops();
    if (flops > 0)
      os << "  maxGflops: " << std::to_string(1.E-9 * static_cast<double>(flops));
    #endif

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
  //! computes a specific moment for the current state
  std::vector<precision> get_moment(moment_id id) const;
  //! computes a specific moment for the current state
  std::vector<precision> get_moment_level(moment_id id) const;
  //! computes and returns the electric field for the current state
  std::vector<precision> get_electric() const;

  //! allows an auxiliary field to be saved for post-processing
  void add_aux_field(aux_field_entry<precision> f) {
    aux_fields.emplace_back(std::move(f));
    if (aux_fields.back().grid.empty()) // if grid provided
      aux_fields.back().grid = grid.get_cells(); // assume the current grid
    if (aux_fields.back().num_dimensions == -1) // default num-dims is the current
      aux_fields.back().num_dimensions = grid.num_dims();
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
  //! synchronizes the grid across MPI ranks and GPU devices
  void grid_sync() {
    #ifdef ASGARD_USE_MPI
    grid.mpi_sync(terms.resources, grid_synced_gen_);
    grid_synced_gen_ = grid.generation();
    #endif
    #ifdef ASGARD_USE_GPU
    grid.gpu_sync();
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

  //! recomputes the moments with the current state, if groupid is negative all groups will be computed
  void compute_moments(group_id gid = group_id{all_groups}) const {
    compute_moments(gid, state);
  }
  //! recomputes the moments given the state of interest and this term group
  void compute_moments(group_id gid, std::vector<precision> const &f) const {
    rassert(terms.moms, "no moments set for this PDE");
    #ifdef ASGARD_USE_MPI
    if (terms.resources.num_ranks() > 1) {
      if (is_leader()) {
        terms.resources.template bcast <precision, resource_comm::regular>(f);
        terms.moms.cache_moments(grid, f, gid.gid);
      } else {
        terms.mpiwork.resize(grid.num_indexes() * hier.block_size());
        terms.resources.template bcast <precision, resource_comm::regular>(terms.mpiwork);
        terms.moms.cache_moments(grid, terms.mpiwork, gid.gid);
      }
    } else {
    #endif
      terms.moms.cache_moments(grid, f, gid.gid);
    #ifdef ASGARD_USE_MPI
    }
    #endif

    compute_poisson(gid);
    terms.rebuild_moment_terms(gid.gid, grid, conn, hier);
  }
  //! recomputes the moments given the state of interest and this term group
  void compute_moments(std::vector<precision> const &f) const {
    compute_moments(group_id{all_groups}, f);
  }
  //! recomputes the Poisson term for the given group
  void compute_poisson(group_id gid) const {
    if (not poisson or (gid.gid >= 0 and not terms.has_poisson(gid.gid)))
      return;

    #ifdef ASGARD_USE_MPI
    // leader must always communicate, the rest only if they have a poisson term
    if (not is_leader() and not terms.has_poisson())
      return;
    #endif

    // currently we only support 1d in position space, so the solver is trivial
    // the cost is so low, that everyone can do it even if it is repeated work
    // when we get to multi-d Poisson problems, the leader will be needed
    // to help the communication process
    poisson.solve_periodic(terms.moms.get_cached_level(poisson.moment0(), hier),
                           terms.moms.edit_poisson_level());
  }
  //! (testing/debugging) copy ns to the current state, e.g., force an initial condition
  void set_current_state(std::vector<precision> const &ns) {
    rassert(ns.size() == state.size(), "cannot set state with different size");
    state = ns;
  }
  //! (debugging) prints the term-matrices
  void print_mats() const;

  #ifdef ASGARD_USE_MPI
  //! returns true if this is mpi rank 0, always true when MPI is not enabled
  bool is_leader() const { return terms.resources.is_leader(); }
  #else
  static constexpr bool is_leader() { return true; }
  #endif

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
  // tag indicating the use of all groups
  static constexpr int all_groups = -1;
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
  void ode_rhs_base(int gid, precision time, std::vector<precision> const &current,
                    std::vector<precision> &R) const;
  //! computes next = current + scale_terms * F(x) + scale_src * sources(time)
  void ode_euler_base(int gid, precision time, std::vector<precision> const &current,
                      terms_scale term_scal, sources_scale source_scal,
                      std::vector<precision> &next) const;
  //! template version of ode right-hand-side sources
  template<data_mode mode>
  void ode_rhs_sources(int gid, precision time, precision alpha, std::vector<precision> &src) const;
  //! returns a snapshot of the state on the current MPI rank
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
  #ifdef ASGARD_USE_MPI
  //! worker iteration apply
  void mpi_iteration_apply_base(int gid, std::vector<precision> &work) const;
  //! leader iteration apply
  void mpi_leader_apply_base(int gid, precision alpha, precision const x[],
                             precision beta, precision y[]) const;
  #endif
#endif // __ASGARD_DOXYGEN_SKIP_INTERNAL

private:
  // indicates the level of noise pushed to the cout
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
  // mutable std::optional<moments1d<precision>> moms1d;
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
