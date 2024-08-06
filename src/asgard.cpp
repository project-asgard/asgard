#include "asgard.hpp"

namespace asgard
{

template<typename precision>
void simulate(std::unique_ptr<PDE<precision>> &pde)
{
  rassert(!!pde, "invalid pde object");
  auto const &options = pde->options();

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

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  node_out() << "  generating: adaptive grid..." << '\n';

  int const degree = options.degree.value();

  adapt::distributed_grid adaptive_grid(*pde);
  node_out() << "  degrees of freedom: "
             << adaptive_grid.size() * fm::ipow(degree + 1, pde->num_dims())
             << '\n';

  node_out() << "  generating: basis operator..." << '\n';
  auto const quiet = false;
  basis::wavelet_transform<precision, resource::host> const
      transformer(*pde, quiet);

  // -- generate and store the mass matrices for each dimension
  node_out() << "  generating: dimension mass matrices..." << '\n';
  generate_dimension_mass_mat<precision>(*pde, transformer);

  // -- generate initial condition vector
  node_out() << "  generating: initial conditions..." << '\n';
  auto initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer);
  node_out() << "  degrees of freedom (post initial adapt): "
                     << adaptive_grid.size() * fm::ipow(degree + 1, pde->num_dims())
                     << '\n';

  // -- regen mass mats after init conditions - TODO: check dims/rechaining?
  generate_dimension_mass_mat<precision>(*pde, transformer);

  // -- generate and store coefficient matrices.
  node_out() << "  generating: coefficient matrices..." << '\n';
  generate_all_coefficients_max_level<precision>(*pde, transformer);

  // -- initialize moments of the PDE
  node_out() << "  generating: moment vectors..." << '\n';
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // this is to bail out for further profiling/development on the setup routines
  if (options.num_time_steps < 1)
    return;

  node_out() << "--- begin time loop staging ---" << '\n';

// -- setup realspace transform for file io or for plotting
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  int dense_size = 0;
  if (options.realspace_output_freq and options.realspace_output_freq.value() > 0)
  {
    dense_size = dense_space_size(*pde);
    expect(dense_size > 0);
  }
  fk::vector<precision> real_space(dense_size);

  // temporary workspaces for the transform
  fk::vector<precision, mem_type::owner, resource::host>
      workspace(dense_size * 2);
  std::array<
      fk::vector<precision, mem_type::view, resource::host>,
      2>
      tmp_workspace = {fk::vector<precision, mem_type::view,
                                          resource::host>(
                           workspace, 0, dense_size - 1),
                       fk::vector<precision, mem_type::view,
                                          resource::host>(
                           workspace, dense_size, dense_size * 2 - 1)};

  // transform initial condition to realspace
  if (options.realspace_output_freq and options.realspace_output_freq.value() > 0)
  {
    wavelet_to_realspace<precision>(*pde, initial_condition,
                                    adaptive_grid.get_table(), transformer,
                                    tmp_workspace, real_space);
  }
#endif

#ifdef ASGARD_USE_MATLAB
  using namespace asgard::ml;
  auto &ml_plot = matlab_plot::get_instance();
  ml_plot.connect(cli_input.get_ml_session_string());
  node_out() << "  connected to MATLAB" << '\n';

  fk::vector<precision> analytic_solution_realspace(dense_size);
  if (pde->has_analytic_soln)
  {
    // generate the analytic solution at t=0
    auto const analytic_solution_init = sum_separable_funcs(
        pde->exact_vector_funcs, pde->get_dimensions(), adaptive_grid,
        transformer, degree, precision{0.0});
    // transform analytic solution to realspace
    wavelet_to_realspace<precision>(
        *pde, analytic_solution_init, adaptive_grid.get_table(), transformer,
        tmp_workspace, analytic_solution_realspace);
  }

  ml_plot.init_plotting(*pde, adaptive_grid.get_table());

  // send initial condition to matlab
  std::vector<size_t> sizes(pde->num_dims);
  for (int i = 0; i < pde->num_dims; i++)
  {
    sizes[i] = (pde->get_dimensions()[i].get_degree() + 1) *
               fm::two_raised_to(pde->get_dimensions()[i].get_level());
  }
  ml_plot.set_var("initial_condition",
                  ml_plot.create_array(sizes, initial_condition));

  ml_plot.copy_pde(*pde);
#endif

  // -- setup output file and write initial condition
  int start_step = 0;
#ifdef ASGARD_IO_HIGHFIVE
  if (not options.restart_file.empty())
  {
    restart_data<precision> data = read_output(
        *pde, adaptive_grid.get_table(), options.restart_file);
    initial_condition = std::move(data.solution);
    start_step        = data.step_index;

    adaptive_grid.recreate_table(data.active_table);

    generate_dimension_mass_mat<precision>(*pde, transformer);
    generate_all_coefficients<precision>(*pde, transformer);
  }
  else
  {
    // compute the realspace moments for the initial file write
    generate_initial_moments(*pde, adaptive_grid, transformer,
                             initial_condition);
  }
  if (options.wavelet_output_freq and options.wavelet_output_freq.value() > 0)
  {
    write_output(*pde, initial_condition,
                 precision{0.0}, 0, initial_condition.size(),
                 adaptive_grid.get_table(), "asgard_wavelet");
  }
  if (options.realspace_output_freq and options.realspace_output_freq.value() > 0)
  {
    write_output(*pde, real_space, precision{0.0}, 0,
                 initial_condition.size(), adaptive_grid.get_table(),
                 "asgard_real");
  }
#endif

  // -- time loop

  fk::vector<precision> f_val(initial_condition);
  node_out() << "--- begin time loop w/ dt " << pde->get_dt()
                     << " ---\n";

  kron_operators<precision> operator_matrices;

  for (auto i = start_step; i < options.num_time_steps.value(); ++i)
  {
    // take a time advance step
    auto const time          = i * pde->get_dt();
    auto const update_system = i == 0;
    auto const method        = options.step_method.value();

    const char *time_str      = "time_advance";
    const std::string time_id = tools::timer.start(time_str);
    f_val                     = time_advance::adaptive_advance(
        method, *pde, operator_matrices, adaptive_grid, transformer,
        f_val, time, update_system);

    tools::timer.stop(time_id);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln())
    {
      // get analytic solution at time(step+1)
      auto const analytic_solution = sum_separable_funcs(
          pde->exact_vector_funcs(), pde->get_dimensions(), adaptive_grid,
          transformer, degree, time + pde->get_dt());

      // calculate root mean squared error
      auto const RMSE = fm::rmserr(f_val, analytic_solution);
      auto const relative_error = 100 * RMSE  / fm::nrminf(analytic_solution);
      auto const [rmse_errors, relative_errors] =
          gather_errors<precision>(RMSE, relative_error);
      expect(rmse_errors.size() == relative_errors.size());
      for (int j = 0; j < rmse_errors.size(); ++j)
      {
        node_out() << "Errors for local rank: " << j << '\n';
        node_out() << "RMSE (numeric-analytic) [wavelet]: "
                   << rmse_errors(j) << '\n';
        node_out()
            << "Relative difference (numeric-analytic) [wavelet]: "
            << relative_errors(j) << " %" << '\n';
      }

#ifdef ASGARD_USE_MATLAB
      if (opts.should_plot(i))
      {
        auto transform_wksp = update_transform_workspace<precision>(
            dense_size, workspace, tmp_workspace);
        if (dense_size > analytic_solution_realspace.size())
        {
          analytic_solution_realspace.resize(dense_size);
        }
        wavelet_to_realspace<precision>(
            *pde, analytic_solution, adaptive_grid.get_table(), transformer,
            transform_wksp, analytic_solution_realspace);
      }
#endif
    }
    else
    {
      node_out() << "No analytic solution found." << '\n';
    }
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)
    /* transform from wavelet space to real space */
    if (pde->is_routput_step(i))
    {
      // resize transform workspaces if grid size changed due to adaptivity
      dense_size          = dense_space_size(*pde);
      auto transform_wksp = update_transform_workspace<precision>(
          dense_size, workspace, tmp_workspace);
      real_space = fk::vector<precision>(dense_size);

      wavelet_to_realspace<precision>(*pde, f_val, adaptive_grid.get_table(),
                                      transformer, transform_wksp,
                                      real_space);
    }
#endif

#ifdef ASGARD_IO_HIGHFIVE
    if (pde->is_output_step(i))
    {
      write_output(*pde, f_val, time + pde->get_dt(), i + 1,
                   f_val.size(), adaptive_grid.get_table(), "asgard_wavelet");
    }
    if (i == options.num_time_steps.value() - 1 and not options.outfile.empty())
    {
      write_output(*pde, f_val, time + pde->get_dt(), i + 1,
                   f_val.size(), adaptive_grid.get_table(), "", options.outfile);
    }
#endif

#ifdef ASGARD_USE_MATLAB
    if (opts.should_plot(i))
    {
      ml_plot.push(std::string("rSpace_" + std::to_string(i)), real_space);

      ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                        analytic_solution_realspace);

      // only plot pde params if the pde has them
      if (parameter_manager<precision>::get_instance().get_num_parameters() >
          0)
      {
        // vlasov pde params plot
        auto dim   = pde->get_dimensions()[0];
        auto nodes = ml_plot.generate_nodes(degree, dim.get_level(),
                                            dim.domain_min, dim.domain_max);

        // evaluates the given PDE parameter at each node
        auto eval_over_nodes = [](std::string const name,
                                  fk::vector<precision> const &nodes_in)
            -> fk::vector<precision> {
          fk::vector<precision> result(nodes_in.size());
          auto param = param_manager.get_parameter(name);
          std::transform(
              nodes_in.begin(), nodes_in.end(), result.begin(),
              [param](precision const &x) { return param->value(x, 0.0); });
          return result;
        };

        fk::vector<precision> n_nodes  = eval_over_nodes("n", nodes);
        fk::vector<precision> u_nodes  = eval_over_nodes("u", nodes);
        fk::vector<precision> th_nodes = eval_over_nodes("theta", nodes);

        // call the matlab script to plot n, u, theta
        ml_plot.reset_params();
        std::vector<size_t> const dim_sizes{1,
                                            static_cast<size_t>(nodes.size())};
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, n_nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, u_nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, th_nodes);
        ml_plot.add_param(time + pde->get_dt());
        ml_plot.call("vlasov_params");
      }
    }
#endif

    node_out() << "timestep: " << i << " complete" << '\n';
  }

  node_out() << "--- simulation complete ---" << '\n';

  auto const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  int my_rank = 0;
#ifdef ASGARD_USE_MPI
  int status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  expect(status == 0);
#endif

  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);

  node_out() << tools::timer.report() << '\n';

#ifdef ASGARD_USE_MATLAB
  ml_plot.close();
#endif
}

#ifdef ASGARD_ENABLE_DOUBLE
template void simulate(std::unique_ptr<PDE<double>> &);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void simulate(std::unique_ptr<PDE<float>> &);
#endif

} // namespace asgard
