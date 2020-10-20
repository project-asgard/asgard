#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "timer.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "boundary_conditions.hpp"
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
  // -- parse cli
  parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }
  options const opts(cli_input);

  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

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

  // -- create forward/reverse mapping between elements and indices
  node_out() << "  generating: element table..." << '\n';

  auto const table = elements::table(opts, *pde);

  node_out() << "  degrees of freedom: "
             << table.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  node_out() << "  generating: basis operator..." << '\n';
  bool const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde,
                                                                   quiet);

  // -- get distribution plan - dividing element grid into subgrids
  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(get_rank());

  // -- generate initial condition vector
  node_out() << "  generating: initial conditions..." << '\n';

  fk::vector<prec> const initial_condition = [&pde, &table, &transformer,
                                              &subgrid, degree]() {
    std::vector<vector_func<prec>> v_functions;

    for (dimension<prec> const &dim : pde->get_dimensions())
    {
      v_functions.push_back(dim.initial_condition);
    }

    return transform_and_combine_dimensions(*pde, v_functions, table,
                                            transformer, subgrid.col_start,
                                            subgrid.col_stop, degree);
  }();

  // -- generate source vectors
  // these will be scaled later according to the simulation time applied
  // with their own time-scaling functions
  node_out() << "  generating: source vectors..." << '\n';
  std::vector<fk::vector<prec>> const initial_sources =
      [&pde, &table, &transformer, &subgrid, degree]() {
        std::vector<fk::vector<prec>> initial_sources;

        for (source<prec> const &source : pde->sources)
        {
          initial_sources.push_back(transform_and_combine_dimensions(
              *pde, source.source_funcs, table, transformer, subgrid.row_start,
              subgrid.row_stop, degree));
        }
        return initial_sources;
      }();

  // -- generate analytic solution vector.
  node_out() << "  generating: analytic solution at t=0 ..." << '\n';

  fk::vector<prec> const analytic_solution = [&pde, &table, &transformer,
                                              &subgrid, degree]() {
    if (pde->has_analytic_soln)
    {
      return transform_and_combine_dimensions(
          *pde, pde->exact_vector_funcs, table, transformer, subgrid.col_start,
          subgrid.col_stop, degree);
    }
    else
    {
      return fk::vector<prec>();
    }
  }();

  // -- generate and store coefficient matrices.

  node_out() << "  generating: coefficient matrices..." << '\n';
  generate_all_coefficients<prec>(*pde, transformer);

  /* generate boundary condition vectors */
  /* these will be scaled later similarly to the source vectors */
  node_out() << "  generating: boundary condition vectors..." << '\n';

  std::array<unscaled_bc_parts<prec>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(
          *pde, table, transformer, subgrid.row_start, subgrid.row_stop);

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
  static int const default_workspace_MB = 10000;

  // FIXME currently used to check realspace transform only
  /* RAM on fusiont5 */
  static int const default_workspace_cpu_MB = 187000;

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE

  // initialize wavelet output
  auto output_dataset = initialize_output_file(initial_condition);

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  int const real_space_size = real_solution_size(*pde);
  fk::vector<prec> real_space(real_space_size);

  // temporary workspaces for the transform
  fk::vector<prec, mem_type::owner, resource::host> workspace(real_space_size *
                                                              2);
  std::array<fk::vector<prec, mem_type::view, resource::host>, 2>
      tmp_workspace = {
          fk::vector<prec, mem_type::view, resource::host>(workspace, 0,
                                                           real_space_size),
          fk::vector<prec, mem_type::view, resource::host>(
              workspace, real_space_size, real_space_size * 2 - 1)};
  // transform initial condition to realspace
  wavelet_to_realspace<prec>(*pde, initial_condition, table, transformer,
                             default_workspace_cpu_MB, tmp_workspace,
                             real_space);

  // initialize realspace output
  std::string const realspace_output_name = "asgard_realspace";
  auto output_dataset_real =
      initialize_output_file(real_space, "asgard_realspace");
#endif

  // -- time loop

  fk::vector<prec> f_val(initial_condition);
  node_out() << "--- begin time loop w/ dt " << pde->get_dt() << " ---\n";
  for (int i = 0; i < opts.num_time_steps; ++i)
  {
    prec const time = i * pde->get_dt();

    if (opts.use_implicit_stepping)
    {
      bool const update_system = i == 0;

      auto const time_id = timer::record.start("implicit_time_advance");
      f_val =
          implicit_time_advance(*pde, table, initial_sources, unscaled_parts,
                                f_val, plan, time, opts.solver, update_system);
      timer::record.stop(time_id);
    }
    else
    {
      // FIXME fold initial sources/unscaled parts into pde object
      // after pde spec/pde split
      auto const &time_id = timer::record.start("explicit_time_advance");
      f_val = explicit_time_advance(*pde, table, opts, initial_sources,
                                    unscaled_parts, f_val, plan,
                                    default_workspace_MB, time);

      timer::record.stop(time_id);
    }

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      prec const time_multiplier = pde->exact_time((i + 1) * pde->get_dt());

      fk::vector<prec> const analytic_solution_t =
          analytic_solution * time_multiplier;
      fk::vector<prec> const diff = f_val - analytic_solution_t;
      prec const RMSE             = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        prec const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();
      auto const relative_error = RMSE / inf_norm(analytic_solution_t) * 100;
      auto const [rmse_errors, relative_errors] =
          gather_errors(RMSE, relative_error);
      assert(rmse_errors.size() == relative_errors.size());
      for (int i = 0; i < rmse_errors.size(); ++i)
      {
        node_out() << "Errors for local rank: " << i << '\n';
        node_out() << "RMSE (numeric-analytic) [wavelet]: " << rmse_errors(i)
                   << '\n';
        node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                   << relative_errors(i) << " %" << '\n';
      }
    }

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
    if (opts.should_output_wavelet(i))
    {
      update_output_file(output_dataset, f_val);
    }

    /* transform from wavelet space to real space */
    if (opts.should_output_realspace(i))
    {
      wavelet_to_realspace<prec>(*pde, f_val, table, transformer,
                                 default_workspace_cpu_MB, tmp_workspace,
                                 real_space);

      update_output_file(output_dataset_real, real_space,
                         realspace_output_name);
    }
#else
    ignore(default_workspace_cpu_MB);
#endif

    node_out() << "timestep: " << i << " complete" << '\n';
  }

  node_out() << "--- simulation complete ---" << '\n';

  int const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  auto const final_result = gather_results(f_val, plan, my_rank, segment_size);

  node_out() << timer::record.report() << '\n';

  finalize_distribution();

  return 0;
}
