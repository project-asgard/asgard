#include "asgard_discretization.hpp"

namespace asgard
{

template<typename precision>
discretization_manager<precision>::discretization_manager(
    std::unique_ptr<PDE<precision>> &&pde_in, verbosity_level verbosity)
    : verb(verbosity), pde(std::move(pde_in)), grid(*pde), transformer(*pde, verb),
      degree_(0), dt_(0), time_(0), time_step_(0), final_time_step_(0),
      kronops(verbosity)
{
  rassert(!!pde, "invalid pde object");

  auto const &options = pde->options();

  if (high_verbosity())
  {
    node_out() << "Branch: " << GIT_BRANCH << '\n';
    node_out() << "Commit Summary: " << GIT_COMMIT_HASH
                    << GIT_COMMIT_SUMMARY << '\n';
    node_out() << "This executable was built on " << BUILD_TIME << '\n';

#ifdef ASGARD_IO_HIGHFIVE
    if (not options.restart_file.empty())
    {
        node_out() << "--- restarting from a file ---\n";
        node_out() << "  filename: " << options.restart_file << '\n';
    }
    else if (get_local_rank() == 0)
        std::cout << options;
#else
    if (get_local_rank() == 0)
        std::cout << options;
#endif

    node_out() << "--- begin setup ---" << '\n';
  }

  // initialize the discretization variables
  degree_ = options.degree.value();

  dt_ = pde->get_dt();

  final_time_step_ = options.num_time_steps.value();

  if (high_verbosity())
    node_out() << "  degrees of freedom: " << degrees_of_freedom() << '\n';

  // -- generate and store the mass matrices for each dimension
  if (high_verbosity())
    node_out() << "  generating: dimension mass matrices..." << '\n';

  generate_dimension_mass_mat(*pde, transformer);

  if (high_verbosity())
    node_out() << "  generating: initial conditions..." << '\n';

  auto initial_condition =
      grid.get_initial_condition(*pde, transformer);

  if (high_verbosity())
    node_out() << "  degrees of freedom (post initial adapt): "
               << degrees_of_freedom() << '\n';

  // WHY IS THIS NEEDED AGAIN!?!?!
  //generate_dimension_mass_mat(*pde, transformer);

  generate_all_coefficients_max_level(*pde, transformer);

#ifdef KRON_MODE_GLOBAL_BLOCK
  if (pde->interp_initial())
  {
    kronops.make(imex_flag::unspecified, *pde, grid);
    vector2d<precision> const &nodes = kronops.get_inodes();
    std::vector<precision> icn(initial_condition.size());
    pde->interp_initial()(nodes, icn);
    fk::vector<precision> ic;
    kronops.get_project(icn.data(), ic);

    for (auto i : indexof(ic))
      initial_condition[i] += ic[i];
  }
#endif

  state = initial_condition.to_std();

  auto const msg = grid.get_subgrid(get_rank());
  fixed_bc = boundary_conditions::make_unscaled_bc_parts(
        *pde, grid.get_table(), transformer, msg.row_start, msg.row_stop);

  if (high_verbosity())
    node_out() << "  generating: moment vectors..." << '\n';

  if (not pde->initial_moments.empty())
  {
    moments.reserve(pde->initial_moments.size());
    for (auto &minit : pde->initial_moments)
      moments.emplace_back(minit);

    for (auto &m : moments)
    {
      m.createFlist(*pde);
      expect(m.get_fList().size() > 0);

      m.createMomentVector(*pde, grid.get_table());
      expect(m.get_vector().size() > 0);
    }
  }

  if (options.step_method.value() == time_advance::method::imex)
    reset_moments();

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE
  if (not options.restart_file.empty())
  {
    restart_data<precision> data = read_output(
        *pde, grid.get_table(), moments, options.restart_file);
    initial_condition = std::move(data.solution);
    time_step_        = data.step_index;

    grid.recreate_table(data.active_table);

    generate_dimension_mass_mat<precision>(*pde, transformer);
    generate_all_coefficients<precision>(*pde, transformer);
  }
  else
  {
    // compute the realspace moments for the initial file write
    generate_initial_moments(*pde, moments, grid, transformer,
                             initial_condition);
  }
  if (options.wavelet_output_freq and options.wavelet_output_freq.value() > 0)
  {
    write_output(*pde, moments, initial_condition,
                 precision{0.0}, 0, initial_condition.size(),
                 grid.get_table(), "asgard_wavelet");
  }
#endif
}

template<typename precision>
void discretization_manager<precision>::save_snapshot(std::filesystem::path const &filename) const
{
#ifdef ASGARD_IO_HIGHFIVE
  fk::vector<precision> fstate(state);
  write_output(*pde, moments, fstate, time_, time_step_, fstate.size(),
               grid.get_table(), "", filename);
#else
  ignore(filename);
  throw std::runtime_error("save_snapshot() requires CMake option -DASGARD_IO_HIGHFIVE=ON");
#endif
}

template<typename precision>
void discretization_manager<precision>::checkpoint() const
{
#ifdef ASGARD_IO_HIGHFIVE
  if (pde->is_output_step(time_step_))
  {
    if (high_verbosity())
      node_out() << "  checkpointing at step = " << time_step_
                  << " (time = " << time_ << ")\n";

    write_output<precision>(*pde, moments, state, time_, time_step_,
                            state.size(), grid.get_table(), "asgard_wavelet");
  }
#endif
}

template<typename precision>
std::optional<std::array<std::vector<precision>, 2>>
discretization_manager<precision>::rmse_exact_sol() const
{
  if (not pde->has_analytic_soln() and not pde->interp_exact())
    return {};

  if (pde->has_analytic_soln())
  {
    fk::vector<precision> fstate(state);

    fk::vector<precision> solution = sum_separable_funcs(
          pde->exact_vector_funcs(), pde->get_dimensions(), grid, transformer,
          degree_, time_);

    // calculate root mean squared error
    auto const RMSE = fm::rmserr(fstate, solution);
    auto const relative_error = 100 * RMSE  / fm::nrminf(solution);
    return gather_errors<precision>(RMSE, relative_error);
  }
  else
  {
#ifdef KRON_MODE_GLOBAL_BLOCK
    vector2d<precision> const &inodes = kronops.get_inodes();
    std::vector<precision> u_exact(inodes.num_strips());
    pde->interp_exact()(time_, inodes, u_exact);

    std::vector<precision> u_comp = kronops.get_nodals(state.data());

    auto const RMSE = fm::rmserr(u_comp, u_exact);
    auto const relative_error = 100 * RMSE  / fm::nrminf(u_exact);
    return gather_errors<precision>(RMSE, relative_error);
#endif
    return {};
  }
}
template<typename precision>
fk::vector<precision>
discretization_manager<precision>::current_mpistate() const
{
  auto const s = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  int my_rank = 0;
#ifdef ASGARD_USE_MPI
  int status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  expect(status == 0);
#endif

  return gather_results<precision>(state, grid.get_distrib_plan(), my_rank, s);
}

#ifdef ASGARD_ENABLE_DOUBLE
template class discretization_manager<double>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class discretization_manager<float>;
#endif

} // namespace asgard
