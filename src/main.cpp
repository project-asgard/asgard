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

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

  // -- parse cli
  parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }
  options const opts(cli_input);

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    finalize_distribution();
    return 0;
  }

  node_out() << "Branch: " << GIT_BRANCH << '\n';
  node_out() << "Commit Summary: " << GIT_COMMIT_HASH << GIT_COMMIT_SUMMARY
             << '\n';
  node_out() << "This executable was built on " << BUILD_TIME << '\n';

  // -- generate pde
  node_out() << "generating: pde..." << '\n';
  auto pde = make_PDE<prec>(cli_input);

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
  auto const degree = pde->get_dimensions()[0].get_degree();

  node_out() << "ASGarD problem configuration:" << '\n';
  node_out() << "  selected PDE: " << cli_input.get_pde_string() << '\n';
  node_out() << "  degree: " << degree << '\n';
  node_out() << "  N steps: " << opts.num_time_steps << '\n';
  node_out() << "  write freq: " << opts.wavelet_output_freq << '\n';
  node_out() << "  realspace freq: " << opts.realspace_output_freq << '\n';
  node_out() << "  implicit: " << opts.use_implicit_stepping << '\n';
  node_out() << "  full grid: " << opts.use_full_grid << '\n';
  node_out() << "  CFL number: " << cli_input.get_cfl() << '\n';
  node_out() << "  Poisson solve: " << opts.do_poisson_solve << '\n';
  node_out() << "  starting levels: ";
  node_out() << std::accumulate(
                    pde->get_dimensions().begin(), pde->get_dimensions().end(),
                    std::string(),
                    [](std::string const &accum, dimension<prec> const &dim) {
                      return accum + std::to_string(dim.get_level()) + " ";
                    })
             << '\n';
  node_out() << "  max adaptivity levels: " << opts.max_level << '\n';

  node_out() << "--- begin setup ---" << '\n';

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  node_out() << "  generating: adaptive grid..." << '\n';

  adapt::distributed_grid adaptive_grid(*pde, opts);
  node_out() << "  degrees of freedom: "
             << adaptive_grid.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  node_out() << "  generating: basis operator..." << '\n';
  auto const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde,
                                                                   quiet);

  // -- generate and store the mass matrices for each dimension
  node_out() << "  generating: dimension mass matrices..." << '\n';
  generate_dimension_mass_mat<prec>(*pde, transformer);

  // -- generate initial condition vector
  node_out() << "  generating: initial conditions..." << '\n';
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);
  node_out() << "  degrees of freedom (post initial adapt): "
             << adaptive_grid.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  // -- regen mass mats after init conditions - TODO: check dims/rechaining?
  generate_dimension_mass_mat<prec>(*pde, transformer);

  // -- generate and store coefficient matrices.
  node_out() << "  generating: coefficient matrices..." << '\n';
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  node_out() << "--- begin time loop staging ---" << '\n';

  // Our default device workspace size is 10GB - 12 GB DRAM on TitanV
  // - a couple GB for allocations not currently covered by the
  // workspace limit (including working batch).

  // This limit is only for the device workspace - the portion
  // of our allocation that will be resident on an accelerator
  // if the code is built for that.
  //
  // FIXME eventually going to be settable from the cmake
  static auto const default_workspace_MB = 10000;

  // FIXME currently used to check realspace transform only
  /* RAM on fusiont5 */
  static auto const default_workspace_cpu_MB = 187000;

// -- setup realspace transform for file io or for plotting
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  auto const real_space_size = real_solution_size(*pde);
  fk::vector<prec> real_space(real_space_size);

  // temporary workspaces for the transform
  fk::vector<prec, mem_type::owner, resource::host> workspace(real_space_size *
                                                              2);
  std::array<fk::vector<prec, mem_type::view, resource::host>, 2>
      tmp_workspace = {
          fk::vector<prec, mem_type::view, resource::host>(workspace, 0,
                                                           real_space_size - 1),
          fk::vector<prec, mem_type::view, resource::host>(
              workspace, real_space_size, real_space_size * 2 - 1)};
  // transform initial condition to realspace
  wavelet_to_realspace<prec>(*pde, initial_condition, adaptive_grid.get_table(),
                             transformer, default_workspace_cpu_MB,
                             tmp_workspace, real_space);
#endif

#ifdef ASGARD_USE_MATLAB
  ml::matlab_plot ml_plot;
  ml_plot.connect(cli_input.get_ml_session_string());
  node_out() << "  connected to MATLAB" << '\n';

  fk::vector<prec> analytic_solution_realspace(real_space_size);
  if (pde->has_analytic_soln)
  {
    // generate the analytic solution at t=0
    auto const subgrid_init           = adaptive_grid.get_subgrid(get_rank());
    auto const analytic_solution_init = transform_and_combine_dimensions(
        *pde, pde->exact_vector_funcs, adaptive_grid.get_table(), transformer,
        subgrid_init.col_start, subgrid_init.col_stop, degree);
    // transform analytic solution to realspace
    wavelet_to_realspace<prec>(
        *pde, analytic_solution_init, adaptive_grid.get_table(), transformer,
        default_workspace_cpu_MB, tmp_workspace, analytic_solution_realspace);
  }

  // Add the matlab scripts directory to the matlab path
  ml_plot.add_param(std::string(ASGARD_SCRIPTS_DIR) + "matlab/");
  ml_plot.call("addpath");

  ml_plot.init_plotting(*pde, adaptive_grid.get_table());
  ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                    analytic_solution_realspace);

  // send initial condition to matlab
  std::vector<size_t> sizes(pde->num_dims);
  for (int i = 0; i < pde->num_dims; i++)
  {
    sizes[i] = pde->get_dimensions()[i].get_degree() *
               fm::two_raised_to(pde->get_dimensions()[i].get_level());
  }
  ml_plot.set_var("initial_condition",
                  ml_plot.create_array(sizes, initial_condition));

  ml_plot.copy_pde(*pde);
#endif

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE
  // initialize wavelet output
  auto output_dataset = initialize_output_file(initial_condition);

  // initialize realspace output
  auto const realspace_output_name = "asgard_realspace";
  auto output_dataset_real =
      initialize_output_file(real_space, "asgard_realspace");
#endif

  // -- time loop

  fk::vector<prec> f_val(initial_condition);
  node_out() << "--- begin time loop w/ dt " << pde->get_dt() << " ---\n";
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    auto const time          = (i + 1) * pde->get_dt();
    auto const update_system = i == 0;
    auto const method   = opts.use_implicit_stepping ? time_advance::method::imp
                                                     : time_advance::method::exp;
    auto const time_str = opts.use_implicit_stepping ? "implicit_time_advance"
                                                     : "explicit_time_advance";
    auto const time_id  = tools::timer.start(time_str);
    auto const sol      = time_advance::adaptive_advance(
        method, *pde, adaptive_grid, transformer, opts, f_val, time,
        default_workspace_MB, update_system);
    f_val.resize(sol.size()) = sol;
    tools::timer.stop(time_id);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      // get analytic solution at time(step+1)
      auto const subgrid           = adaptive_grid.get_subgrid(get_rank());
      auto const time_multiplier   = pde->exact_time(time + pde->get_dt());
      auto const analytic_solution = transform_and_combine_dimensions(
          *pde, pde->exact_vector_funcs, adaptive_grid.get_table(), transformer,
          subgrid.col_start, subgrid.col_stop, degree, time, time_multiplier);

      // calculate root mean squared error
      auto const diff = f_val - analytic_solution;
      auto const RMSE = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        auto const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();
      auto const relative_error = RMSE / inf_norm(analytic_solution) * 100;
      auto const [rmse_errors, relative_errors] =
          gather_errors(RMSE, relative_error);
      expect(rmse_errors.size() == relative_errors.size());
      for (int j = 0; j < rmse_errors.size(); ++j)
      {
        node_out() << "Errors for local rank: " << j << '\n';
        node_out() << "RMSE (numeric-analytic) [wavelet]: " << rmse_errors(j)
                   << '\n';
        node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                   << relative_errors(j) << " %" << '\n';
      }

#ifdef ASGARD_USE_MATLAB
      if (opts.should_plot(i))
      {
        auto const real_size = real_solution_size(*pde);
        auto transform_wksp  = update_transform_workspace<prec>(
            real_size, workspace, tmp_workspace);
        if (real_size > analytic_solution_realspace.size())
        {
          analytic_solution_realspace.resize(real_size);
        }
        wavelet_to_realspace<prec>(*pde, analytic_solution,
                                   adaptive_grid.get_table(), transformer,
                                   default_workspace_cpu_MB, transform_wksp,
                                   analytic_solution_realspace);
      }
#endif
    }
    else
    {
      node_out() << "No analytic solution found." << '\n';
    }
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)
    /* transform from wavelet space to real space */
    if (opts.should_output_realspace(i) || opts.should_plot(i))
    {
      // resize transform workspaces if grid size changed due to adaptivity
      auto const real_size = real_solution_size(*pde);
      auto transform_wksp =
          update_transform_workspace<prec>(real_size, workspace, tmp_workspace);
      real_space.resize(real_size);

      wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                 transformer, default_workspace_cpu_MB,
                                 transform_wksp, real_space);
    }
#endif

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
    if (opts.should_output_wavelet(i))
    {
      update_output_file(output_dataset, f_val);
    }
    if (opts.should_output_realspace(i))
    {
      update_output_file(output_dataset_real, real_space,
                         realspace_output_name);
    }
#else
    ignore(default_workspace_cpu_MB);
#endif

#ifdef ASGARD_USE_MATLAB
    if (opts.should_plot(i))
    {
      ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                        analytic_solution_realspace);
    }
#endif

    node_out() << "timestep: " << i << " complete" << '\n';
  }

  node_out() << "--- simulation complete ---" << '\n';

  auto const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);

  node_out() << tools::timer.report() << '\n';

  finalize_distribution();

  return 0;
}
