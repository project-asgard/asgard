#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
void discretization_manager<precision>::start_cold(pde_scheme<precision> &pde)
{
  conn = connection_patterns(pde.max_level());

  int const degree_ = options_.degree.value();

  if (high_verbosity()) {
    std::cout << '\n';
    #ifdef ASGARD_HAS_GITINFO
    std::cout << "ASGarD: git-branch '" << ASGARD_GIT_BRANCH << "'\n";
    std::cout << "  " << ASGARD_GIT_COMMIT_HASH << ASGARD_GIT_COMMIT_SUMMARY << '\n';
    std::cout << " -- discretization options --\n";
    #else
    std::cout << " -- ASGarD release " << ASGARD_RELEASE_INFO << '\n';
    #endif
  } else {
    if (not stop_verbosity())
      std::cout << "\n -- ASGarD discretization options --\n";
  }

  grid = sparse_grid(options_);
  #ifdef ASGARD_USE_GPU
  grid.gpu_sync();
  #endif

  if (not stop_verbosity()) {
    if (not options_.title.empty())
      std::cout << "    title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "           " << options_.subtitle << '\n';

    std::cout << "basis degree: " << degree_to_string(degree_) << '\n';

    std::cout << grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_relative)
      std::cout << "  relative tolerance: " << options_.adapt_relative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_relative)
      std::cout << "  non-adaptive\n";
  }

  { // setting up the time-step approach
    // if no method is set, defaulting to explicit time-stepping
    time_method sm = options_.step_method.value_or(time_method::rk3);

    time_data dtime; // initialize below

    precision stop = options_.stop_time.value_or(-1);
    precision dt   = options_.dt.value_or(-1);
    int64_t n      = options_.num_time_steps.value_or(-1);

    if (sm == time_method::steady) {
      stop  = options_.stop_time.value_or(options_.default_stop_time.value_or(0));
      dtime = time_data(stop);
    } else {
      rassert(not (stop >= 0 and dt >= 0 and n >= 0),
        "Must provide exactly two of the three time-stepping parameters: -dt, -num-steps, -time");

      // replace options with defaults, when appropriate
      if (n == 0 or stop == 0) { // initial conditions only, no time stepping
        n = 0;
        dt = 0;
        stop = -1; // ignore stop below
      } else if (n > 0) {
        if (stop < 0 and dt < 0) {
          dt = options_.default_dt.value_or(-1);
          if (dt < 0) {
            stop = options_.default_stop_time.value_or(-1);
            if (stop < 0)
              throw std::runtime_error("number of steps provided, but no dt or stop-time");
          }
        }
      } else if (stop >= 0) { // no num-steps, but dt may be provided or have a default
        if (dt < 0) {
          dt = options_.default_dt.value_or(-1);
          if (dt < 0)
            throw std::runtime_error("stop-time provided but no time-step or number of steps");
        }
      } else if (dt >= 0) { // both n and stop are unspecified
        stop = options_.default_stop_time.value_or(-1);
        if (stop < 0)
          throw std::runtime_error("dt provided, but no stop-time or number of steps");
      } else { // nothing provided, look for defaults
        dt   = options_.default_dt.value_or(-1);
        stop = options_.default_stop_time.value_or(-1);
        if (dt < 0 or stop < 0)
          throw std::runtime_error("need at least two time parameters: -dt, -num-steps, -time");
      }

      if (n >= 0 and stop >= 0 and dt < 0)
        dtime = time_data(sm, n, time_data::input_stop_time{stop});
      else if (dt >= 0 and stop >= 0 and n < 0)
        dtime = time_data(sm, time_data::input_dt{dt},
                          time_data::input_stop_time{stop});
      else if (dt >= 0 and n >= 0 and stop < 0)
        dtime = time_data(sm, time_data::input_dt{dt}, n);
      else
        throw std::runtime_error("how did this happen?");
    }

    if (is_imex(sm)) {
      stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
    } else {
      stepper = time_advance_manager<precision>(dtime, options_);
    }
  }

  if (not stop_verbosity())
    std::cout << stepper;

  if (stepper.needs_solver() and not options_.solver)
    throw std::runtime_error("the selected time-stepping method requires a solver, "
                             "or a default solver set in the pde specification");

  hier = hierarchy_manipulator(degree_, domain_);

  // first we must initialize the terms, which will also initialize the kron
  // operations and the interpolation engine
  terms = term_manager<precision>(options_, domain_, pde, grid, hier, conn);

  set_initial_condition();

  if (not stop_verbosity()) {
    int64_t const dof = grid.num_indexes() * hier.block_size();
    std::cout << "initial degrees of freedom: " << tools::split_style(dof) << "\n\n";
  }

  start_moments(); // grid may have changes above, wait to start the moments

  terms.build_matrices(grid, conn, hier);

  if (high_verbosity())
    progress_report();
}

template<typename precision>
void discretization_manager<precision>::restart_from_file(pde_scheme<precision> &pde)
{
#ifdef ASGARD_USE_HIGHFIVE
  if (not stop_verbosity())
    std::cout << "restarting from file: \"" << options_.restart_file << "\"\n";

  tools::time_event timing_("restart from file");

  time_data dtime;
  h5manager<precision>::read(options_.restart_file, high_verbosity(),
                             options_, domain_, grid,
                             dtime, aux_fields, state);

  conn = connection_patterns(options_.max_level());

  #ifdef ASGARD_USE_GPU
  grid.gpu_sync();
  #endif

  hier = hierarchy_manipulator(options_.degree.value(), domain_);

  if (is_imex(dtime.step_method())) {
    stepper = time_advance_manager<precision>(dtime, options_, pde.imex_im(), pde.imex_ex());
  } else {
    stepper = time_advance_manager<precision>(dtime, options_);
  }

  terms = term_manager<precision>(options_, domain_, pde, grid, hier, conn);

  start_moments();

  terms.build_matrices(grid, conn, hier);

  if (not stop_verbosity()) {
    if (not options_.title.empty())
      std::cout << "  title: " << options_.title << '\n';
    if (not options_.subtitle.empty())
      std::cout << "subtitle: " << options_.subtitle << '\n';

    std::cout << "basis degree: " << degree_to_string(hier.degree()) << '\n';

    std::cout << grid;
    if (options_.adapt_threshold)
      std::cout << "  adaptive tolerance: " << options_.adapt_threshold.value() << '\n';
    if (options_.adapt_relative)
      std::cout << "  relative tolerance: " << options_.adapt_relative.value() << '\n';
    if (not options_.adapt_threshold and not options_.adapt_relative)
      std::cout << "  non-adaptive\n";
    std::cout << stepper;
    if (high_verbosity())
      progress_report();
  }

#else
  ignore(pde);
  throw std::runtime_error("restarting from a file requires CMake option "
                           "-DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::start_moments() {
  if (terms.moms)
    compute_moments(state);

  if (terms.has_poisson()) {
    moment_id const m0 = terms.moms.find_id(moment::zero(domain_.num_vel()));
    poisson = solvers::poisson(degree(), domain_.xleft(0), domain_.xright(0),
                               grid.current_level(0), m0);
  }
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const {
#ifdef ASGARD_USE_HIGHFIVE
  #ifdef ASGARD_USE_MPI
  if (not is_leader())
    return;
  #endif
  h5manager<precision>::write(options_, domain_, degree(), grid, stepper.data,
                              state, terms.moms, aux_fields, filename);
#else
  ignore(filename);
  throw std::runtime_error("saving to a file requires CMake option -DASGARD_USE_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::set_initial_condition()
{
  precision const atol = options_.adapt_threshold.value_or(0);
  precision const rtol = options_.adapt_relative.value_or(0);

  #ifdef ASGARD_USE_MPI
  if (not is_leader()) {
    this->grid_sync();
    state.resize(grid.num_indexes() * hier.block_size());
    terms.prapare_kron_workspace(grid);
    return;
  }
  #endif

  bool keep_refining = true;

  constexpr precision time = 0;

  int iterations = 0;
  while (keep_refining)
  {
    state.resize(grid.num_indexes() * hier.block_size());

    if (initial_md_)
      terms.interp(grid, conn, time, 1, initial_md_, 0, state, terms.kwork, terms.it1, terms.it2);
    else
      std::fill(state.begin(), state.end(), precision{0});

    for (int i : iindexof(initial_sep_)) {
      expect(initial_sep_[i].num_dims() == num_dims());

      terms.rebuild_mass_matrices(grid);

      std::array<block_diag_matrix<precision>, max_num_dimensions> mock;

      hier.template project_separable<data_mode::increment>
            (initial_sep_[i], grid, terms.lmass, time, 1, state.data());
    }

    if (atol > 0 or rtol > 0) {
      // on the first iteration, do both refine and coarsen with a full-adapt
      // on follow-on iteration, only add more nodes for stability and to avoid stagnation
      sparse_grid::strategy mode = (iterations == 0) ? sparse_grid::strategy::adapt
                                                     : sparse_grid::strategy::refine;
      int const gid = grid.generation();
      grid.refine(atol, rtol, hier.block_size(), conn[connect_1d::hierarchy::volume], mode, state);

      // if the grid remained the same, there's nothing to do
      keep_refining = (gid != grid.generation());

      if (keep_refining) // should only do this if using interpolation, otherwise just do at the end
        terms.prapare_kron_workspace(grid);

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

  terms.rebuild_mass_matrices(grid);
  for (int i : iindexof(sep)) {
    hier.template project_separable<data_mode::increment>
          (sep[i], grid, terms.lmass, time, 1, out.data());
  }
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_moment(moment_id id) const {
  std::vector<precision> result;
  terms.moms.compute(grid, id, state, result);
  return result;
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_moment_level(moment_id id) const {
  rassert(domain_.num_pos() == 1, "level completion is done only for 1 position dimension");
  std::vector<precision> tmp;
  std::vector<precision> result;
  terms.moms.compute(grid, id, state, tmp);
  terms.moms.complete_level(hier, tmp, result);
  return result;
}

template<typename precision>
std::vector<precision> discretization_manager<precision>::get_electric() const {
  rassert(poisson, "get_electric() requires a PDE with terms with electric dependence");
  terms.moms.cache_moment(poisson.moment0(), grid, state);
  poisson.solve_periodic(terms.moms.get_cached_level(poisson.moment0(), hier),
                         terms.moms.edit_poisson_level());
  return terms.moms.poisson_level();
}

template<typename precision>
void discretization_manager<precision>::print_mats() const {
  int const num_dims = terms.num_dims;
  for (auto tid : iindexof(terms.terms)) {
    for (int d : iindexof(num_dims)) {
      std::cout << " term = " << tid << "  dim = " << d << '\n';
      if (terms.terms[tid].coeffs[d].empty()) {
        std::cout << "identity\n";
      } else {
        terms.terms[tid].coeffs[d].to_full(conn).print(std::cout);
      }
      std::cout << '\n';
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//        source and terms apply methods
///////////////////////////////////////////////////////////////////////////////
template<typename precision>
void discretization_manager<precision>::ode_rhs_base(
    int gid, precision time, std::vector<precision> const &x, std::vector<precision> &y) const
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
  // the effective input vector, in MPI context this is either mpiwork or R
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

  // locally update all moments
  if (terms.moms)
    compute_moments(group_id{gid}, in);

  out.resize(in.size());

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(gid, grid, conn);
    tools::time_event performance_("ode-rhs kronmult", flops);
    #else
    tools::time_event performance_("ode-rhs kronmult");
    #endif
    terms.apply(gid, grid, conn, -1, in, 0, out);

    if (not terms.has_terms()) // R wasn't zeroes out above
        std::fill(y.begin(), y.end(), 0);
  }{
    tools::time_event performance_("ode-rhs sources");
    terms.template apply_sources<data_mode::increment>(gid, grid, conn, hier, time, 1, out);
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
    int gid, precision time, std::vector<precision> const &current,
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

  // locally update all moments
  if (terms.moms)
    compute_moments(group_id{gid}, in);

  {
    #ifdef ASGARD_USE_FLOPCOUNTER
    int64_t const flops = terms.flop_count(gid, grid, conn);
    tools::time_event performance_("ode-rhs kronmult", flops);
    #else
    tools::time_event performance_("ode-rhs kronmult");
    #endif
    if (is_leader())
      out = in;
    else
      out.resize(in.size());

    if (term_scal.value != 0)
      terms.apply(gid, grid, conn, -term_scal.value, in, (is_leader()) ? 1 : 0, out);

    if (not terms.has_terms()) // R wasn't zeroes out above
        std::fill(out.begin(), out.end(), 0);
  }{
    tools::time_event performance_("ode-rhs sources");
    if (source_scal.value == 1)
      terms.template apply_sources<data_mode::increment>(gid, grid, conn, hier, time, 1, out);
    else
      terms.template apply_sources<data_mode::scal_inc>(gid, grid, conn, hier, time,
                                                        source_scal.value, out);
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
    int gid, precision time, precision alpha, std::vector<precision> &src) const {
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
      terms.template apply_sources<mode>(gid, grid, conn, hier, time, alpha, terms.mpiwork);
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
      terms.template apply_sources<mm>(gid, grid, conn, hier, time, alpha, src);
      terms.resources.reduce_add(src);
    }
  } else {
  #endif
    terms.template apply_sources<mode>(gid, grid, conn, hier, time, alpha, src);
  #ifdef ASGARD_USE_MPI
  }
  #endif
}

#ifdef ASGARD_USE_MPI
template<typename precision>
void discretization_manager<precision>::mpi_iteration_apply_base(
    int gid, std::vector<precision> &y) const
{
  rassert(not is_leader(), "cannot call mpi_iteration_apply() on the leader rank");

  tools::time_event performance_("terms-apply");

  y.resize(grid.num_indexes() * hier.block_size());

  std::vector<precision> &x = terms.mpiwork;
  x.resize(y.size());

  while (true) // will break-exist from the loop
  {
    terms.resources.bcast(x); // get the input from the leader

    // if the last entry is equal to the numeric-max, stop
    // the numeric max is the "kill" signal, since it will not happen in a real run
    if (x.back() == std::numeric_limits<precision>::max())
      break;

    terms.apply(gid, grid, conn, 1, x, 0, y);

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
  w.resize(grid.num_indexes() * hier.block_size());
  w.back() = std::numeric_limits<precision>::max();
  terms.resources.bcast(w);
}
template<typename precision>
void discretization_manager<precision>::mpi_leader_apply_base(
    int gid, precision alpha, precision const x[], precision beta, precision y[]) const
{
  rassert(is_leader(), "mpi_leader_apply() can be called only on the leader rank");
  tools::time_event performance_("mpi_leader_apply");

  if (terms.resources.num_ranks() == 1) {
    terms.apply(gid, grid, conn, alpha, x, beta, y);
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

  terms.apply(gid, grid, conn, 1, x, 0, work.data());

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
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;

template void discretization_manager<double>::ode_rhs_sources<data_mode::increment>(
    int, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::scal_inc>(
    int, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::replace>(
    int, double, double, std::vector<double> &) const;
template void discretization_manager<double>::ode_rhs_sources<data_mode::scal_rep>(
    int, double, double, std::vector<double> &) const;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;

template void discretization_manager<float>::ode_rhs_sources<data_mode::increment>(
    int, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::scal_inc>(
    int, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::replace>(
    int, float, float, std::vector<float> &) const;
template void discretization_manager<float>::ode_rhs_sources<data_mode::scal_rep>(
    int, float, float, std::vector<float> &) const;
#endif

} // namespace asgard
