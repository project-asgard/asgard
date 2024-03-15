#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

#include "asgard_vector.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

using prec = asgard::default_precision;

int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = asgard::initialize_distribution();

  // -- parse cli
  asgard::parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    asgard::node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }
  asgard::options const opts(cli_input);

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    asgard::finalize_distribution();
    return 0;
  }

  asgard::node_out() << "Branch: " << GIT_BRANCH << '\n';
  asgard::node_out() << "Commit Summary: " << GIT_COMMIT_HASH
                     << GIT_COMMIT_SUMMARY << '\n';
  asgard::node_out() << "This executable was built on " << BUILD_TIME << '\n';

  // -- generate pde
  asgard::node_out() << "generating: pde..." << '\n';
  auto pde = asgard::make_PDE<prec>(cli_input);

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
  auto const degree = pde->get_dimensions()[0].get_degree();

  asgard::node_out() << "ASGarD problem configuration:" << '\n';
  asgard::node_out() << "  selected PDE: " << cli_input.get_pde_string()
                     << '\n';
  asgard::node_out() << "  degree: " << degree << '\n';
  asgard::node_out() << "  N steps: " << opts.num_time_steps << '\n';
  asgard::node_out() << "  write freq: " << opts.wavelet_output_freq << '\n';
  asgard::node_out() << "  realspace freq: " << opts.realspace_output_freq
                     << '\n';
  asgard::node_out() << "  implicit: " << opts.use_implicit_stepping << '\n';
  asgard::node_out() << "  full grid: " << opts.use_full_grid << '\n';
  asgard::node_out() << "  CFL number: " << cli_input.get_cfl() << '\n';
  asgard::node_out() << "  Poisson solve: " << opts.do_poisson_solve << '\n';
  asgard::node_out() << "  starting levels: ";
  asgard::node_out() << std::accumulate(
                            pde->get_dimensions().begin(),
                            pde->get_dimensions().end(), std::string(),
                            [](std::string const &accum,
                               asgard::dimension<prec> const &dim) {
                              return accum + std::to_string(dim.get_level()) +
                                     " ";
                            })
                     << '\n';
  asgard::node_out() << "  max adaptivity levels: " << opts.max_level << '\n';

  asgard::node_out() << "--- begin setup ---" << '\n';

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  asgard::node_out() << "  generating: adaptive grid..." << '\n';

  asgard::adapt::distributed_grid adaptive_grid(*pde, opts);
  asgard::node_out() << "  degrees of freedom: "
                     << adaptive_grid.size() * static_cast<uint64_t>(std::pow(
                                                   degree, pde->num_dims))
                     << '\n';

  asgard::node_out() << "  generating: basis operator..." << '\n';
  auto const quiet = false;
  asgard::basis::wavelet_transform<prec, asgard::resource::host> const
      transformer(opts, *pde, quiet);

  // -- generate and store the mass matrices for each dimension
  asgard::node_out() << "  generating: dimension mass matrices..." << '\n';
  asgard::generate_dimension_mass_mat<prec>(*pde, transformer);

  // -- generate initial condition vector
  asgard::node_out() << "  generating: initial conditions..." << '\n';
  auto initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);
  asgard::node_out() << "  degrees of freedom (post initial adapt): "
                     << adaptive_grid.size() * static_cast<uint64_t>(std::pow(
                                                   degree, pde->num_dims))
                     << '\n';

  // -- regen mass mats after init conditions - TODO: check dims/rechaining?
  asgard::generate_dimension_mass_mat<prec>(*pde, transformer);

  // -- generate and store coefficient matrices.
  asgard::node_out() << "  generating: coefficient matrices..." << '\n';
  asgard::generate_all_coefficients_max_level<prec>(*pde, transformer);

  // -- initialize moments of the PDE
  asgard::node_out() << "  generating: moment vectors..." << '\n';
  for (auto &m : pde->moments)
  {
    m.createFlist(*pde, opts);
    expect(m.get_fList().size() > 0);

    m.createMomentVector(*pde, opts, adaptive_grid.get_table());
    expect(m.get_vector().size() > 0);
  }

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  asgard::node_out() << "--- begin time loop staging ---" << '\n';

// -- setup realspace transform for file io or for plotting
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  int dense_size = 0;
  if (cli_input.get_realspace_output_freq() > 0)
  {
    dense_size = asgard::dense_space_size(*pde);
    expect(dense_size > 0);
  }
  asgard::fk::vector<prec> real_space(dense_size);

  // temporary workspaces for the transform
  asgard::fk::vector<prec, asgard::mem_type::owner, asgard::resource::host>
      workspace(dense_size * 2);
  std::array<
      asgard::fk::vector<prec, asgard::mem_type::view, asgard::resource::host>,
      2>
      tmp_workspace = {asgard::fk::vector<prec, asgard::mem_type::view,
                                          asgard::resource::host>(
                           workspace, 0, dense_size - 1),
                       asgard::fk::vector<prec, asgard::mem_type::view,
                                          asgard::resource::host>(
                           workspace, dense_size, dense_size * 2 - 1)};
  // transform initial condition to realspace
  if (cli_input.get_realspace_output_freq() > 0)
  {
    asgard::wavelet_to_realspace<prec>(*pde, initial_condition,
                                       adaptive_grid.get_table(), transformer,
                                       tmp_workspace, real_space);
  }
#endif

#ifdef ASGARD_USE_MATLAB
  using namespace asgard::ml;
  auto &ml_plot = matlab_plot::get_instance();
  ml_plot.connect(cli_input.get_ml_session_string());
  asgard::node_out() << "  connected to MATLAB" << '\n';

  asgard::fk::vector<prec> analytic_solution_realspace(dense_size);
  if (pde->has_analytic_soln)
  {
    // generate the analytic solution at t=0
    auto const analytic_solution_init = sum_separable_funcs(
        pde->exact_vector_funcs, pde->get_dimensions(), adaptive_grid,
        transformer, degree, static_cast<prec>(0.0));
    // transform analytic solution to realspace
    asgard::wavelet_to_realspace<prec>(
        *pde, analytic_solution_init, adaptive_grid.get_table(), transformer,
        tmp_workspace, analytic_solution_realspace);
  }

  ml_plot.init_plotting(*pde, adaptive_grid.get_table());
  ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                    analytic_solution_realspace);

  // send initial condition to matlab
  std::vector<size_t> sizes(pde->num_dims);
  for (int i = 0; i < pde->num_dims; i++)
  {
    sizes[i] = pde->get_dimensions()[i].get_degree() *
               asgard::fm::two_raised_to(pde->get_dimensions()[i].get_level());
  }
  ml_plot.set_var("initial_condition",
                  ml_plot.create_array(sizes, initial_condition));

  ml_plot.copy_pde(*pde);
#endif

  // -- setup output file and write initial condition
  int start_step = 0;
#ifdef ASGARD_IO_HIGHFIVE
  if (cli_input.do_restart())
  {
    asgard::restart_data<prec> data = asgard::read_output(
        *pde, adaptive_grid.get_table(), cli_input.get_restart_file());
    initial_condition = std::move(data.solution);
    start_step        = data.step_index;

    adaptive_grid.recreate_table(data.active_table, data.max_level);

    asgard::generate_dimension_mass_mat<prec>(*pde, transformer);
    asgard::generate_all_coefficients<prec>(*pde, transformer);
  }
  else
  {
    // compute the realspace moments for the initial file write
    asgard::generate_initial_moments(*pde, opts, adaptive_grid, transformer,
                                     initial_condition);
  }
  if (cli_input.get_wavelet_output_freq() > 0)
  {
    asgard::write_output(*pde, cli_input, initial_condition,
                         static_cast<prec>(0.0), 0, initial_condition.size(),
                         adaptive_grid.get_table(), "asgard_wavelet");
  }
  if (cli_input.get_realspace_output_freq() > 0)
  {
    asgard::write_output(*pde, cli_input, real_space, static_cast<prec>(0.0), 0,
                         initial_condition.size(), adaptive_grid.get_table(),
                         "asgard_real");
  }
#endif

  // -- time loop

  asgard::fk::vector<prec> f_val(initial_condition);
  asgard::node_out() << "--- begin time loop w/ dt " << pde->get_dt()
                     << " ---\n";

  asgard::matrix_list<prec> operator_matrices;

  for (auto i = start_step; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    auto const time          = (i + 1) * pde->get_dt();
    auto const update_system = i == 0;
    auto const method =
        opts.use_implicit_stepping
            ? asgard::time_advance::method::imp
            : (opts.use_imex_stepping ? asgard::time_advance::method::imex
                                      : asgard::time_advance::method::exp);
    const char *time_str =
        opts.use_implicit_stepping
            ? "implicit_time_advance"
            : (opts.use_imex_stepping ? "imex_time_advance"
                                      : "explicit_time_advance");
    const std::string time_id = asgard::tools::timer.start(time_str);
    f_val                     = asgard::time_advance::adaptive_advance(
        method, *pde, operator_matrices, adaptive_grid, transformer, opts,
        f_val, time, update_system);

    asgard::tools::timer.stop(time_id);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      // get analytic solution at time(step+1)
      auto const analytic_solution = sum_separable_funcs(
          pde->exact_vector_funcs, pde->get_dimensions(), adaptive_grid,
          transformer, degree, time + pde->get_dt());

      // calculate root mean squared error
      auto const diff = f_val - analytic_solution;
      auto const RMSE = [&diff]() {
        asgard::fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        auto const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();
      auto const relative_error =
          RMSE / asgard::inf_norm(analytic_solution) * 100;
      auto const [rmse_errors, relative_errors] =
          asgard::gather_errors<prec>(RMSE, relative_error);
      expect(rmse_errors.size() == relative_errors.size());
      for (int j = 0; j < rmse_errors.size(); ++j)
      {
        asgard::node_out() << "Errors for local rank: " << j << '\n';
        asgard::node_out() << "RMSE (numeric-analytic) [wavelet]: "
                           << rmse_errors(j) << '\n';
        asgard::node_out()
            << "Relative difference (numeric-analytic) [wavelet]: "
            << relative_errors(j) << " %" << '\n';
      }

#ifdef ASGARD_USE_MATLAB
      if (opts.should_plot(i))
      {
        auto transform_wksp = asgard::update_transform_workspace<prec>(
            dense_size, workspace, tmp_workspace);
        if (dense_size > analytic_solution_realspace.size())
        {
          analytic_solution_realspace.resize(dense_size);
        }
        asgard::wavelet_to_realspace<prec>(
            *pde, analytic_solution, adaptive_grid.get_table(), transformer,
            transform_wksp, analytic_solution_realspace);
      }
#endif
    }
    else
    {
      asgard::node_out() << "No analytic solution found." << '\n';
    }
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)
    /* transform from wavelet space to real space */
    if (opts.should_output_realspace(i) || opts.should_plot(i))
    {
      // resize transform workspaces if grid size changed due to adaptivity
      dense_size          = asgard::dense_space_size(*pde);
      auto transform_wksp = asgard::update_transform_workspace<prec>(
          dense_size, workspace, tmp_workspace);
      // real_space.resize(dense_size);
      real_space = asgard::fk::vector<prec>(dense_size);

      asgard::wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                         transformer, transform_wksp,
                                         real_space);
    }
#endif

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
    if (opts.should_output_wavelet(i))
    {
      asgard::write_output(*pde, cli_input, f_val, time, i + 1, f_val.size(),
                           adaptive_grid.get_table(), "asgard_wavelet");
    }
    if (opts.should_output_realspace(i))
    {
      asgard::write_output(*pde, cli_input, real_space, time, i + 1,
                           f_val.size(), adaptive_grid.get_table(),
                           "asgard_real");
    }
#endif

#ifdef ASGARD_USE_MATLAB
    if (opts.should_plot(i))
    {
      ml_plot.push(std::string("rSpace_" + std::to_string(i)), real_space);

      ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                        analytic_solution_realspace);

      // only plot pde params if the pde has them
      if (asgard::parameter_manager<prec>::get_instance().get_num_parameters() >
          0)
      {
        // vlasov pde params plot
        auto dim   = pde->get_dimensions()[0];
        auto nodes = ml_plot.generate_nodes(degree, dim.get_level(),
                                            dim.domain_min, dim.domain_max);

        // evaluates the given PDE parameter at each node
        auto eval_over_nodes = [](std::string const name,
                                  asgard::fk::vector<prec> const &nodes_in)
            -> asgard::fk::vector<prec> {
          asgard::fk::vector<prec> result(nodes_in.size());
          using P    = prec;
          auto param = asgard::param_manager.get_parameter(name);
          std::transform(
              nodes_in.begin(), nodes_in.end(), result.begin(),
              [param](prec const &x) { return param->value(x, 0.0); });
          return result;
        };

        asgard::fk::vector<prec> n_nodes  = eval_over_nodes("n", nodes);
        asgard::fk::vector<prec> u_nodes  = eval_over_nodes("u", nodes);
        asgard::fk::vector<prec> th_nodes = eval_over_nodes("theta", nodes);

        // call the matlab script to plot n, u, theta
        ml_plot.reset_params();
        std::vector<size_t> const dim_sizes{1,
                                            static_cast<size_t>(nodes.size())};
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, n_nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, u_nodes);
        ml_plot.add_param({1, static_cast<size_t>(nodes.size())}, th_nodes);
        ml_plot.add_param(time);
        ml_plot.call("vlasov_params");
      }
    }
#endif

    asgard::node_out() << "timestep: " << i << " complete" << '\n';
  }

  asgard::node_out() << "--- simulation complete ---" << '\n';

  auto const segment_size = asgard::element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);

  asgard::node_out() << asgard::tools::timer.report() << '\n';

  asgard::finalize_distribution();

#ifdef ASGARD_USE_MATLAB
  ml_plot.close();
#endif

  return 0;
}
