#include "asgard_discretization.hpp"

#ifdef ASGARD_USE_GPU
#include "asgard_gpu_algorithms.hpp"
#endif

namespace asgard
{
template<typename precision>
discretization_manager<precision>::discretization_manager(
    pde_scheme<precision> pde, verbosity_level verbosity)
    : discretization_manager()
{
  verb = pde.options().verbosity.value_or(verbosity);

  #ifdef ASGARD_ALWAYS_SAFE_STEP
  safe_step = true;
  #else
  safe_step = pde.options().safe_step;
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

  if (high_verbosity()) {
    std::cout << '\n';
    #ifdef ASGARD_HAS_GITINFO
    std::cout << "ASGarD: git-branch '" << ASGARD_GIT_BRANCH << "'\n";
    std::cout << "  " << ASGARD_GIT_COMMIT_HASH << ASGARD_GIT_COMMIT_SUMMARY << '\n';
    #else
    std::cout << " -- ASGarD release " << ASGARD_RELEASE_INFO << '\n';
    #endif
    int const num_threads = kronmult::get_num_omp_threads();
    if (num_threads > 0) {
      std::cout << "        OpenMP num-threads: " << num_threads << '\n';
    } else {
      std::cout << "        OpenMP: disabled\n";
    }
    #ifdef ASGARD_USE_CUDA
    std::cout << "        GPU Backend: CUDA\n";
    #else
    #ifdef ASGARD_USE_ROCM
    std::cout << "        GPU Backend: ROCM\n";
    #else
    std::cout << "        GPU Backend: disabled\n";
    #endif
    #endif
  }

  if (options_.restarting())
    restart_from_file(pde);
  else
    start_cold(pde);
}

template<typename precision>
void discretization_manager<precision>::start_cold(pde_scheme<precision> &pde)
{
  int const degree_ = options_.degree.value();

  if (not stop_verbosity()) {
    // show general information about the problem
    // indicate that work has started, should hit this point almost instantaneously after launch
    std::cout << "\n -- ASGarD discretization options --\n";
    if (not options_.title.empty())
      std::cout << "    title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "           " << options_.subtitle << '\n';

    std::cout << "basis degree: " << degree_to_string(degree_) << '\n';
  }

  // initialize the terms, which will also initialize the kron and interpolation engines
  // this operation can take some time due to building mass matrices
  terms = term_manager<precision>(options_, domain_, pde, sparse_grid(options_));

  { // setting up the time-stepper
    // reading the from the options, user-selected first, if missing fallback to default options
    time_data const dtime = make_time_data(options_);

    if (is_imex(dtime.step_method())) {
      stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
    } else {
      stepper = time_advance_manager<precision>(dtime, options_);
    }
  }

  if (not stop_verbosity()) {
    // continue the report
    std::cout << terms.grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_relative)
      std::cout << "  relative tolerance: " << options_.adapt_relative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_relative)
      std::cout << "  non-adaptive\n";

    std::cout << stepper;

    #ifndef ASGARD_ALWAYS_SAFE_STEP
    if (safe_step)
      std::cout << "enabled safety checks for invalid floats\n";
    #endif
  }

  if (stepper.needs_solver() and not options_.solver)
    throw std::runtime_error("the selected time-stepping method requires a solver, "
                             "or a default solver set in the pde specification");

  refinement = refinement_manager<precision>(options_, pde);

  // setting the initial conditions uses refinement, must come after the refinement_manager
  // this iterates depending on the adapt-weight and the separable/interpolation conditions
  // this is the first point of potentially heavy work
  set_initial_condition();

  if (not stop_verbosity())
    std::cout << "initial degrees of freedom: " << tools::split_style(terms.num_dof()) << "\n\n";

  start_moments(); // grid may have changes above, wait to start the moments

  terms.build_matrices(); // the matrices may need the moments from above

  if (high_verbosity())
    progress_report();

  if (options_.show_memusage) {
    std::cout << "memory usage after initial setup\n";
    report_memusage();
  }
}

template<typename precision>
void discretization_manager<precision>::restart_from_file(pde_scheme<precision> &pde)
{
#ifdef ASGARD_USE_HIGHFIVE

  if (not stop_verbosity())
    std::cout << "restarting from file: \"" << options_.restart_file << "\"\n";

  tools::time_event timing_("restart from file");

  sparse_grid grid;
  time_data dtime;
  h5manager<precision>::read(options_.restart_file, high_verbosity(),
                             options_, domain_, grid,
                             dtime, aux_fields, state);

  if (is_imex(dtime.step_method())) {
    stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
  } else {
    stepper = time_advance_manager<precision>(dtime, options_);
  }

  // show general problem properties
  if (not stop_verbosity()) {
    if (not options_.title.empty())
      std::cout << "    title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "           " << options_.subtitle << '\n';
  }

  terms = term_manager<precision>(options_, domain_, pde, std::move(grid));

  refinement = refinement_manager<precision>(options_, pde);

  start_moments();

  terms.build_matrices();

  if (not stop_verbosity()) {
    std::cout << "basis degree: " << degree_to_string(terms.degree()) << '\n';

    std::cout << grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_relative)
      std::cout << "  relative tolerance: " << options_.adapt_relative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_relative)
      std::cout << "  non-adaptive\n";
    std::cout << stepper;
    #ifndef ASGARD_ALWAYS_SAFE_STEP
    if (safe_step)
      std::cout << "enabled safety checks for invalid floats\n";
    #endif
    if (high_verbosity())
      progress_report();
  }

  if (options_.show_memusage) {
    std::cout << "memory usage after restart\n";
    report_memusage();
  }

#else
  std::ignore = pde;
  throw std::runtime_error("restarting from a file requires CMake option "
                           "-DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::start_moments() {
  if (terms.moms)
    compute_moments_(group_id::all(), state);

  if (terms.has_poisson()) {
    moment_id const m0 = terms.moms.find_id(moment::zero(domain_.num_vel()));
    poisson = solvers::poisson(degree(), domain_.xleft(0), domain_.xright(0),
                               terms.grid.current_level(0), m0);
  }
}

template<typename precision>
void discretization_manager<precision>::compute_moments_(
    group_id gid, std::vector<precision> const &f) const
{
  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (is_leader()) {
      terms.resources.template bcast <precision, resource_comm::regular>(f);
      compute_moments_local(gid, f);
    } else {
      terms.mpiwork.resize(terms.num_dof());
      terms.resources.template bcast <precision, resource_comm::regular>(terms.mpiwork);
      compute_moments_local(gid, terms.mpiwork);
    }
  } else {
  #endif
    compute_moments_local(gid, f);
  #ifdef ASGARD_USE_MPI
  }
  #endif
}

template<typename precision>
void discretization_manager<precision>::compute_moments_local(
    group_id gid, std::vector<precision> const &f) const
{
  terms.moms.cache_moments(gid, terms.grid, f);
  terms.moms.load_interp(gid, terms.interp, terms.kwork);
  compute_poisson(gid);
  terms.rebuild_moment_terms(gid);
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const {
#ifdef ASGARD_USE_HIGHFIVE
  #ifdef ASGARD_USE_MPI
  if (not is_leader())
    return;
  #endif
  h5manager<precision>::write(options_, domain_, degree(), terms.grid, stepper.data,
                              state, terms.moms, aux_fields, filename);
#else
  std::ignore = filename;
  throw std::runtime_error("saving to a file requires CMake option -DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::save_final_snapshot() const
{
  if (options_.show_memusage) {
    std::cout << "final memory usage\n";
    report_memusage();
  }
  if (not options_.outfile.empty())
    save_snapshot(options_.outfile);
}

template<typename precision>
void discretization_manager<precision>::set_initial_condition()
{
  #ifdef ASGARD_USE_MPI
  if (not is_leader()) {
    this->grid_sync();
    state.resize(terms.num_dof());
    terms.prapare_kron_workspace();
    return;
  }
  #endif

  sparse_grid &grid = terms.grid;

  bool keep_refining = true;

  constexpr precision time = 0;

  int iterations = 0;
  while (keep_refining)
  {
    state.resize(terms.num_dof());

    if (initial_md_)
      terms.interp(grid, terms.conn, {}, time, 1,
                   // using the moment signature, even thought the initial conditions
                   // cannot have a moment dependence
                   [&](precision t, vector2d<precision> const &x,
                       momentset<precision> const &, std::vector<precision> &vals)
                       -> void {
                         initial_md_(t, x, vals);
                   }, 0, state, terms.kwork);
    else
      std::fill(state.begin(), state.end(), precision{0});

    for (int i : iindexof(initial_sep_)) {
      assert(initial_sep_[i].num_dims() == num_dims());

      terms.rebuild_mass_matrices();

      terms.hier.template project_separable<data_mode::increment>
            (initial_sep_[i], grid, terms.lmass, time, 1, state.data());
    }

    if (refinement) {
      // on the first iteration, do both refine and coarsen with a full-adapt
      // on follow-on iteration, we should only add more nodes for stability and to avoid stagnation
      // however, we can also run into issue with refinement instability, e.g., due to moments
      // and adapt-weights, where we over-refine due to inf/nan interpolation weights
      // so every 5 iterations or so, we can drop some of the coefficients
      // should do at least 2 refine iterations for every adapt in order to avoid stagnation
      sparse_grid::strategy mode = (iterations % 3 == 0) ? sparse_grid::strategy::adapt
                                                         : sparse_grid::strategy::refine;

      if (iterations > 2 * terms.max_level) // should not go this far unless stagnating
        mode = sparse_grid::strategy::refine;

      int const gid = grid.generation();
      #ifdef ASGARD_USE_GPU
      gpu::vector<precision> gstate = state;
      refine(mode, gstate);
      #else
      refine(mode, state);
      #endif

      // if the grid remained the same, there's nothing to do
      keep_refining = (gid != grid.generation());

      if (keep_refining) { // should only do this if using interpolation, otherwise just do at the end
        terms.grid.gpu_sync();
        terms.prapare_kron_workspace();
      }

    } else { // no refinement set, use the grid as-is
      keep_refining = false;
    }

    iterations++;
  }

  this->grid_sync();
}

template<typename precision> void
discretization_manager<precision>::project_function(
    std::vector<separable_func<precision>> const &sep,
    md_func<precision> const &, std::vector<precision> &out) const
{
  tools::time_event performance_("project functions");

  if (out.empty())
    out.resize(state.size());
  else {
    out.resize(state.size());
    std::fill(out.begin(), out.end(), 0);
  }

  precision time = stepper.data.time();

  terms.rebuild_mass_matrices();
  for (int i : iindexof(sep)) {
    terms.hier.template project_separable<data_mode::increment>
          (sep[i], terms.grid, terms.lmass, time, 1, out.data());
  }
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_moment(moment_id id) const {
  std::vector<precision> result;
  terms.moms.mcompute(terms.grid, id, state, result);
  return result;
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_moment_level(moment_id id) const {
  rassert(domain_.num_pos() == 1, "level completion is done only for 1 position dimension");
  std::vector<precision> tmp;
  std::vector<precision> result;
  terms.moms.mcompute(terms.grid, id, state, tmp);
  terms.moms.complete_level(terms.hier, tmp, result);
  return result;
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_electric() const {
  rassert(poisson, "get_electric() requires a PDE with terms with electric dependence");
  terms.moms.cache_moment(poisson.moment0(), terms.grid, state);
  poisson.solve_periodic(terms.moms.get_cached_level(poisson.moment0(), terms.hier),
                         terms.moms.edit_poisson_level());
  return terms.moms.poisson_level();
}

template<typename precision>
void discretization_manager<precision>::print_mats() const {
  int const numd = num_dims();
  for (auto tid : iindexof(terms.terms)) {
    for (int d : iindexof(numd)) {
      std::cout << " term = " << tid << "  dim = " << d << '\n';
      if (terms.terms[tid].coeffs[d].empty()) {
        std::cout << "identity\n";
      } else {
        terms.terms[tid].coeffs[d].to_full(terms.conn).print(std::cout);
      }
      std::cout << '\n';
    }
  }
}

template<typename precision>
void discretization_manager<precision>::report_memusage(std::ostream &os) const {
  auto MB = [](size_t bytes) -> std::string {
    std::string s = std::to_string(bytes / (1024 * 1024)) + "MB\n";
    s.insert(0, 11 - s.size(), ' ');
    return s;
  };
  os << "sparse grid " << MB(terms.grid.used_bytes());
  os << "hierarchy   " << MB(terms.hier.used_bytes());
  terms.print_bytes(os);
  stepper.print_bytes(os);
}

///////////////////////////////////////////////////////////////////////////////
//        source and terms apply methods
///////////////////////////////////////////////////////////////////////////////
template<typename precision>
void discretization_manager<precision>::ode_rhs_base(
    group_id group, precision time, std::vector<precision> const &x,
    std::vector<precision> &y) const
{
  // 1. broadcast x to all ranks, then compute the moments
  //    (the moments can be done locally, the work is cheap)
  // 2. apply the terms and sources
  // 3. collect (reduce-add) the result into y
  // naturally, if not using MPI or using only 1 rank, there is no broadcast/reduce
  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    terms.mpiwork.resize(x.size());
    if (is_leader()) {
      terms.resources.bcast(x);
    } else {
      terms.resources.bcast(terms.mpiwork);
    }
  }
  #endif

  // the effective input vector, in MPI context this is either current or mpiwork
  // leader just uses current, the rest use mpiwork
  std::vector<precision> const &in = [&]() -> std::vector<precision> const &
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() == 1 or is_leader())
        return x;
      else
        return terms.mpiwork;
      #else
      return x;
      #endif
    }();
  // the effective output vector, in MPI context this is either mpiwork or R
  std::vector<precision> &out = [&]() -> std::vector<precision> &
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() > 1 and is_leader())
        return terms.mpiwork;
      else
        return y;
      #else
      return y;
      #endif
    }();

  if (terms.moms)
    compute_moments_local(group, in);

  out.resize(in.size());

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(group, grid, conn);
    tools::time_event performance_("ode-rhs terms", flops);
    #else
    tools::time_event performance_("ode-rhs terms");
    #endif
    terms.apply(group, -1, in, 0, out);

    if (not terms.has_terms()) // R wasn't zeroes out above
        std::fill(out.begin(), out.end(), 0);
  }{
    tools::time_event performance_("ode-rhs sources");
    terms.template apply_sources<data_mode::increment>(group, time, 1, out);
  }

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (is_leader())
      terms.resources.reduce_add(out, y);
    else
      terms.resources.reduce_add(out);
  }
  #endif
}

template<typename precision>
void discretization_manager<precision>::ode_euler_base(
    group_id group, precision time, std::vector<precision> const &current,
    terms_scale term_scal, sources_scale source_scal, std::vector<precision> &next) const
{
  // 1. broadcast x to all ranks, then compute the moments
  //    (the moments can be done locally, the work is cheap)
  // 2. apply the terms and sources with the two scales
  // 3. collect (reduce-add) the result into y
  // naturally, if not using MPI or using only 1 rank, there is no broadcast/reduce
  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    terms.mpiwork.resize(current.size());
    if (is_leader()) {
      terms.resources.bcast(current);
    } else {
      terms.resources.bcast(terms.mpiwork);
    }
  }
  #endif

  // the effective input vector, in MPI context this is either current or mpiwork
  // leader just uses current, the rest use mpiwork
  std::vector<precision> const &in = [&]() -> std::vector<precision> const &
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() == 1 or is_leader())
        return current;
      else
        return terms.mpiwork;
      #else
      return current;
      #endif
    }();
  // the effective input vector, in MPI context this is either mpiwork or R
  std::vector<precision> &out = [&]() -> std::vector<precision> &
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() > 1 and is_leader())
        return terms.mpiwork;
      else
        return next;
      #else
      return next;
      #endif
    }();

  if (terms.moms)
    compute_moments_local(group, in);

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(group, grid, conn);
    tools::time_event performance_("ode-rhs kronmult", flops);
    #else
    tools::time_event performance_("ode-rhs kronmult");
    #endif
    if (is_leader())
      out = in;
    else
      out.resize(in.size());

    if (term_scal.value != 0) {
      terms.apply(group, -term_scal.value, in, (is_leader()) ? 1 : 0, out);
      if (not terms.has_terms()) // R wasn't zeroes out above
        std::fill(out.begin(), out.end(), 0);
    } else {
      // term_scal == 0 means ignoring the terms, the leader is already set to in
      // so the workers have to zero out their vectors, others out will be uninitialized
      if (not is_leader())
        std::fill(out.begin(), out.end(), 0);
    }
  }{
    tools::time_event performance_("ode-rhs sources");
    if (source_scal.value == 1)
      terms.template apply_sources<data_mode::increment>(group, time, 1, out);
    else
      terms.template apply_sources<data_mode::scal_inc>(group, time, source_scal.value, out);
  }

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (is_leader())
      terms.resources.reduce_add(out, next);
    else
      terms.resources.reduce_add(out);
  }
  #endif
}

template<typename precision>
template<data_mode mode>
void discretization_manager<precision>::ode_rhs_sources(
    group_id group, precision time, precision alpha, std::vector<precision> &src) const {
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
      terms.template apply_sources<mode>(group, time, alpha, terms.mpiwork);
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
      terms.template apply_sources<mm>(group, time, alpha, src);
      terms.resources.reduce_add(src);
    }
  } else {
  #endif
    terms.template apply_sources<mode>(group, time, alpha, src);
  #ifdef ASGARD_USE_MPI
  }
  #endif
}

#ifdef ASGARD_USE_GPU
template<typename precision>
void discretization_manager<precision>::compute_moments_gpu_(group_id gid, precision const f[]) const {
  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    int64_t const num_entries = terms.num_dof();
    if (is_leader()) {
      terms.resources.template bcast_gpu <precision, resource_comm::regular>(num_entries, f);
      compute_moments_local_gpu(gid, f);
    } else {
      terms.gpumpi_work.resize(num_entries);
      terms.resources.template bcast_gpu <precision, resource_comm::regular>(
            num_entries, terms.gpumpi_work.data());
      compute_moments_local_gpu(gid, terms.gpumpi_work.data());
    }
  } else {
  #endif
    compute_moments_local_gpu(gid, f);
  #ifdef ASGARD_USE_MPI
  }
  #endif
}

template<typename precision>
void discretization_manager<precision>::compute_moments_local_gpu(
    group_id gid, precision const f[]) const
{
  {
    // const-cast is safe here, since wf is only used as "const" in the call
    gpu::wrap_array<precision> wf(const_cast<precision *>(f), num_dof());
    terms.moms.compute_moments(gid, terms.grid, terms.interp, terms.kwork, wf.vec);
  }
  compute_poisson(gid);
  terms.rebuild_moment_terms(gid);
}

template<typename precision>
void discretization_manager<precision>::ode_rhs_base_gpu(
    group_id group, precision time, precision const current[], precision R[]) const
{
  int64_t const num_entries = num_dof();
  #ifdef ASGARD_USE_MPI
  assert(num_entries < static_cast<int64_t>(std::numeric_limits<int>::max()));
  int const inume = static_cast<int>(num_entries);
  if (terms.resources.num_ranks() > 1) {
    terms.gpumpi_work.resize(num_entries);
    if (is_leader()) {
      terms.resources.bcast_gpu(inume, current);
    } else {
      terms.resources.bcast_gpu(inume, terms.gpumpi_work.data());
    }
  }
  #endif

  // the effective input array, in MPI context this is either current or mpiwork
  // leader just uses current, the rest use mpiwork
  precision const *in = [&]() -> precision const *
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() == 1 or is_leader())
        return current;
      else
        return terms.gpumpi_work.data();
      #else
      return current;
      #endif
    }();
  // the effective output array, in MPI context this is either mpiwork or R
  precision *out = [&]() -> precision *
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() > 1 and is_leader())
        return terms.gpumpi_work.data();
      else
        return R;
      #else
      return R;
      #endif
    }();

  if (terms.moms) compute_moments_local_gpu(group, in);

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(group, grid, conn);
    tools::time_event performance_("ode-rhs-gpu terms", flops);
    #else
    tools::time_event performance_("ode-rhs-gpu terms");
    #endif
    terms.apply_gpu(group, -1, in, 0, out);

    if (not terms.has_terms()) // R wasn't zeroes out above
      compute->fill_zeros(num_entries, out);
  }{
    tools::time_event performance_("ode-rhs-gpu sources");
    terms.template apply_sources_gpu<data_mode::increment>(group, time, 1, out);
  }

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (is_leader())
      terms.resources.reduce_add_gpu(inume, out, R);
    else
      terms.resources.reduce_add_gpu(inume, out);
  }
  #endif
}

template<typename precision>
void discretization_manager<precision>::ode_euler_base_gpu(
    group_id group, precision time, precision const current[], terms_scale term_scal,
    sources_scale source_scal, precision next[]) const
{
  int64_t const num_entries = num_dof();
  #ifdef ASGARD_USE_MPI
  assert(num_entries < static_cast<int64_t>(std::numeric_limits<int>::max()));
  int const inume = static_cast<int>(num_entries);
  if (terms.resources.num_ranks() > 1) {
    terms.gpumpi_work.resize(num_entries);
    if (is_leader()) {
      terms.resources.bcast_gpu(inume, current);
    } else {
      terms.resources.bcast_gpu(inume, terms.gpumpi_work.data());
    }
  }
  #endif

  // the effective input vector, in MPI context this is either current or mpiwork
  // leader just uses current, the rest use mpiwork
  precision const *in = [&]() -> precision const *
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() == 1 or is_leader())
        return current;
      else
        return terms.gpumpi_work.data();
      #else
      return current;
      #endif
    }();
  // the effective input vector, in MPI context this is either mpiwork or R
  precision *out = [&]() -> precision *
    {
      #ifdef ASGARD_USE_MPI
      if (terms.resources.num_ranks() > 1 and is_leader())
        return terms.gpumpi_work.data();
      else
        return next;
      #else
      return next;
      #endif
    }();

  if (terms.moms) compute_moments_local_gpu(group, in);

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(group, conn);
    tools::time_event performance_("ode-rhs-gpu terms", flops);
    #else
    tools::time_event performance_("ode-rhs-gpu terms");
    #endif
    if (is_leader()) {
      gpu::memcopy_dev2dev(num_entries, in, out);
    }

    if (term_scal.value != 0) {
      terms.apply_gpu(group, -term_scal.value, in, (is_leader()) ? 1 : 0, out);
      if (not terms.has_terms())
        compute->fill_zeros(num_entries, out);
    } else {
      if (not is_leader())
        compute->fill_zeros(num_entries, out);
    }
  }{
    tools::time_event performance_("ode-rhs-gpu sources");
    if (source_scal.value == 1)
      terms.template apply_sources_gpu<data_mode::increment>(group, time, 1, out);
    else
      terms.template apply_sources_gpu<data_mode::scal_inc>(group, time, source_scal.value, out);
  }

  #ifdef ASGARD_USE_MPI
  if (terms.resources.num_ranks() > 1) {
    if (is_leader())
      terms.resources.reduce_add_gpu(inume, out, next);
    else
      terms.resources.reduce_add_gpu(inume, out);
  }
  #endif
}

template<typename precision>
template<data_mode mode>
void discretization_manager<precision>::ode_rhs_sources_gpu(
    group_id group, precision time, precision alpha, precision src[]) const {
  tools::time_event performance_("ode sources");
  #ifdef ASGARD_USE_MPI
  int64_t const num_entries = num_dof();
  if (terms.resources.num_ranks() > 1) {
    terms.gpumpi_work.resize(num_entries);
    if constexpr (mode == data_mode::replace or mode == data_mode::scal_rep) {
      compute->fill_zeros(terms.gpumpi_work);
    } else {
      gpu::memcopy_dev2dev(num_entries, src, terms.gpumpi_work.data());
    }
    if (is_leader()) {
      terms.template apply_sources_gpu<mode>(group, time, alpha, terms.gpumpi_work.data());
      terms.resources.reduce_add_gpu(num_entries, terms.gpumpi_work.data(), src);
    } else {
      data_mode constexpr mm = [=]()-> data_mode {
          if constexpr (mode == data_mode::increment)
            return data_mode::replace;
          else if constexpr (mode == data_mode::scal_inc)
            return data_mode::scal_rep;
          else
            return mode;
        }();
      terms.template apply_sources_gpu<mm>(group, time, alpha, src);
      terms.resources.reduce_add_gpu(num_entries, src);
    }
  } else {
  #endif
    terms.template apply_sources_gpu<mode>(group, time, alpha, src);
  #ifdef ASGARD_USE_MPI
  }
  #endif
}
#endif

#ifdef ASGARD_USE_MPI
template<typename precision>
void discretization_manager<precision>::mpi_iteration_apply_base(
    group_id group, std::vector<precision> &y) const
{
  rassert(not is_leader(), "cannot call mpi_iteration_apply() on the leader rank");

  tools::time_event performance_("terms-apply");

  y.resize(terms.num_dof());

  std::vector<precision> &x = terms.mpiwork;
  x.resize(y.size());

  while (true) // will break-exist from the loop
  {
    terms.resources.bcast(x); // get the input from the leader

    // if the last entry is equal to the numeric-max, stop
    // the numeric max is the "kill" signal, since it will not happen in a real run
    if (x.back() == std::numeric_limits<precision>::max())
      break;

    terms.apply(group, 1, x, 0, y);

    if (not terms.has_terms()) // R must be zeroed out explicitly
      std::fill(y.begin(), y.end(), 0);

    terms.resources.reduce_add(y);
  }
}
template<typename precision>
void discretization_manager<precision>::mpi_iteration_stop() const
{
  // only the leader calls the "stop" and only non-leader can be stopped
  // make sure the leader is calling and there is someone to call
  if (not is_leader() or terms.resources.num_ranks() == 1)
    return;

  std::vector<precision> &w = terms.mpiwork;
  w.resize(terms.num_dof());
  w.back() = std::numeric_limits<precision>::max();
  terms.resources.bcast(w);
}
template<typename precision>
void discretization_manager<precision>::mpi_leader_apply_base(
    group_id group, precision alpha, precision const x[], precision beta, precision y[]) const
{
  rassert(is_leader(), "mpi_leader_apply() can be called only on the leader rank");
  tools::time_event performance_("mpi_leader_apply");

  if (terms.resources.num_ranks() == 1) {
    terms.apply(group, alpha, x, beta, y);
    return;
  }

  std::vector<precision> &work = terms.mpiwork;

  int const n = static_cast<int>(state_size());
  // each rank computes w = terms * x, if alpha = 1 and beta = 0, then that's the answer
  // using different alpha/beta means obtaining w first, then computing alpha * w + beta * y
  if (alpha == 1 and beta == 0)
    work.resize(n);
  else
    work.resize(2 * n);

  terms.resources.bcast(n, x);

  terms.apply(group, 1, x, 0, work.data());

  if (not terms.has_terms() and beta == 0) // mpiwork must be zeroed out explicitly (??)
    std::fill_n(work.begin(), n, 0);

  if (work.size() == static_cast<size_t>(n)) { // alpha == 1 and beta == 0
    terms.resources.reduce_add(n, work.data(), y);
  } else {
    terms.resources.reduce_add(n, work.data(), work.data() + n);
    ASGARD_OMP_PARFOR_SIMD
    for (size_t i = 0; i < static_cast<size_t>(n); i++)
      y[i] = alpha * work[i + n] + beta * y[i];
  }
}

#ifdef ASGARD_USE_GPU
template<typename precision>
void discretization_manager<precision>::mpi_iteration_apply_base_gpu(
    group_id group, precision y[]) const
{
  rassert(not is_leader(), "cannot call mpi_iteration_apply() on the leader rank");

  tools::time_event performance_("terms-apply");

  int64_t const num_entries = num_dof();
  terms.gpumpi_work.resize(num_entries);
  precision *x = terms.gpumpi_work.data();

  while (true) // will break-exist from the loop
  {
    terms.resources.bcast_gpu(num_entries, x); // get the input from the leader

    // if the last entry is equal to the numeric-max, stop
    // the numeric max is the "kill" signal, since it will not happen in a real run
    // (maybe look for a better way to do this)
    precision back = 0;
    gpu::memcopy_dev2host(1, x + num_entries - 1, &back);
    if (back == std::numeric_limits<precision>::max())
      break;

    terms.apply_gpu(group, 1, x, 0, y);

    if (not terms.has_terms()) // R must be zeroed out explicitly
      compute->fill_zeros(num_entries, x);

    terms.resources.reduce_add_gpu(num_entries, y);
  }
}
template<typename precision>
void discretization_manager<precision>::mpi_iteration_stop_gpu() const
{
  // only the leader calls the "stop" and only non-leader can be stopped
  // make sure the leader is calling and there is someone to call
  if (not is_leader() or terms.resources.num_ranks() == 1)
    return;

  int64_t const num_entries = num_dof();
  terms.gpumpi_work.resize(num_entries);
  precision *x = terms.gpumpi_work.data();

  precision const v = std::numeric_limits<precision>::max();
  gpu::memcopy_host2dev(1, &v, x + num_entries - 1);
  terms.resources.bcast_gpu(num_entries, x);
}
template<typename precision>
void discretization_manager<precision>::mpi_leader_apply_base_gpu(
    group_id group, precision alpha, precision const x[], precision beta, precision y[]) const
{
  rassert(is_leader(), "mpi_leader_apply() can be called only on the leader rank");
  tools::time_event performance_("mpi_leader_apply");

  if (terms.resources.num_ranks() == 1) {
    terms.apply_gpu(group, alpha, x, beta, y);
    return;
  }

  int64_t const num_entries = num_dof();
  terms.gpumpi_work.resize(num_entries);

  // each rank computes w = terms * x, if alpha = 1 and beta = 0, then that's the answer
  // using different alpha/beta means obtaining w first, then computing alpha * w + beta * y
  if (alpha == 1 and beta == 0)
    terms.gpumpi_work.resize(num_entries);
  else
    terms.gpumpi_work.resize(2 * num_entries);

  precision *work = terms.gpumpi_work.data();

  terms.resources.bcast_gpu(num_entries, x);

  terms.apply_gpu(group, 1, x, 0, work);

  if (not terms.has_terms() and beta == 0) // mpiwork must be zeroed out explicitly (??)
    compute->fill_zeros(num_entries, work);

  if (alpha == 1 and beta == 0) { // alpha == 1 and beta == 0
    terms.resources.reduce_add_gpu(num_entries, work, y);
  } else {
    terms.resources.reduce_add_gpu(num_entries, work, work + num_entries);
    gpu::axpby(num_entries, alpha, work + num_entries, beta, y);
  }
}
#endif
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;

template void discretization_manager<double>::ode_rhs_sources<data_mode::increment>(
    group_id, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::scal_inc>(
    group_id, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::replace>(
    group_id, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::scal_rep>(
    group_id, double, double, std::vector<double> &) const;

#ifdef ASGARD_USE_GPU
template void discretization_manager<double>::ode_rhs_sources_gpu<data_mode::increment>(
    group_id, double, double, double[]) const;
template void discretization_manager<double>::ode_rhs_sources_gpu<data_mode::scal_inc>(
    group_id, double, double, double[]) const;
template void discretization_manager<double>::ode_rhs_sources_gpu<data_mode::replace>(
    group_id, double, double, double[]) const;
template void discretization_manager<double>::ode_rhs_sources_gpu<data_mode::scal_rep>(
    group_id, double, double, double[]) const;
#endif

#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;

template void discretization_manager<float>::ode_rhs_sources<data_mode::increment>(
    group_id, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::scal_inc>(
    group_id, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::replace>(
    group_id, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::scal_rep>(
    group_id, float, float, std::vector<float> &) const;

#ifdef ASGARD_USE_GPU
template void discretization_manager<float>::ode_rhs_sources_gpu<data_mode::increment>(
    group_id, float, float, float[]) const;
template void discretization_manager<float>::ode_rhs_sources_gpu<data_mode::scal_inc>(
    group_id, float, float, float[]) const;
template void discretization_manager<float>::ode_rhs_sources_gpu<data_mode::replace>(
    group_id, float, float, float[]) const;
template void discretization_manager<float>::ode_rhs_sources_gpu<data_mode::scal_rep>(
    group_id, float, float, float[]) const;
#endif

#endif

} // namespace asgard
