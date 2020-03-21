#include "batch.hpp"
#include "build_info.hpp"
#include "chunk.hpp"
#include "coefficients.hpp"
#include "connectivity.hpp"
#include "distribution.hpp"
#include "element_table.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

using prec = double;
int main(int argc, char **argv)
{
  int unused = 0;
  options opts(argc, argv);
  if (!opts.is_valid())
  {
    node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }

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

  // -- parse user input and generate pde
  node_out() << "generating: pde..." << '\n';
  auto pde = make_PDE<prec>(opts.get_selected_pde(), opts.get_level(),
                            opts.get_degree());

  // sync up options object in case pde defaults were loaded
  // assume uniform level and degree across dimensions
  opts.update_level(pde->get_dimensions()[0].get_level());
  opts.update_degree(pde->get_dimensions()[0].get_degree());

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
  auto const degree = pde->get_dimensions()[0].get_degree();

  node_out() << "ASGarD problem configuration:" << '\n';
  node_out() << "  selected PDE: " << opts.get_pde_string() << '\n';
  node_out() << "  level: " << opts.get_level() << '\n';
  node_out() << "  degree: " << opts.get_degree() << '\n';
  node_out() << "  N steps: " << opts.get_time_steps() << '\n';
  node_out() << "  write freq: " << opts.get_write_frequency() << '\n';
  node_out() << "  realspace freq: " << opts.get_realspace_output_freq()
             << '\n';
  node_out() << "  implicit: " << opts.using_implicit() << '\n';
  node_out() << "  full grid: " << opts.using_full_grid() << '\n';
  node_out() << "  CFL number: " << opts.get_cfl() << '\n';
  node_out() << "  Poisson solve: " << opts.do_poisson_solve() << '\n';

  node_out() << "--- begin setup ---" << '\n';

  // -- create forward/reverse mapping between elements and indices
  node_out() << "  generating: element table..." << '\n';
  element_table const table = element_table(opts, pde->num_dims);

  node_out() << "  degrees of freedom: "
             << table.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  // -- get distribution plan - dividing element grid into subgrids
  auto const plan    = get_plan(num_ranks, table);
  auto const subgrid = plan.at(get_rank());

  // -- generate initial condition vector.
  node_out() << "  generating: initial conditions..." << '\n';

  fk::vector<prec> const initial_condition = [&pde, &table, &subgrid,
                                              degree]() {
    std::vector<vector_func<prec>> v_functions;

    for (dimension<prec> const &dim : pde->get_dimensions())
    {
      v_functions.push_back(dim.initial_condition);
    }

    return transform_and_combine_dimensions(
        *pde, v_functions, table, subgrid.col_start, subgrid.col_stop, degree);
  }();

  // -- generate source vectors.
  // these will be scaled later according to the simulation time applied
  // with their own time-scaling functions
  node_out() << "  generating: source vectors..." << '\n';
  std::vector<fk::vector<prec>> const initial_sources = [&pde, &table, &subgrid,
                                                         degree]() {
    std::vector<fk::vector<prec>> initial_sources;

    for (source<prec> const &source : pde->sources)
    {
      initial_sources.push_back(transform_and_combine_dimensions(
          *pde, source.source_funcs, table, subgrid.row_start, subgrid.row_stop,
          degree));
    }
    return initial_sources;
  }();

  // -- generate analytic solution vector.
  node_out() << "  generating: analytic solution at t=0 ..." << '\n';

  fk::vector<prec> const analytic_solution = [&pde, &table, &subgrid,
                                              degree]() {
    if (pde->has_analytic_soln)
    {
      return transform_and_combine_dimensions(*pde, pde->exact_vector_funcs,
                                              table, subgrid.col_start,
                                              subgrid.col_stop, degree);
    }
    else
    {
      return fk::vector<prec>();
    }
  }();

  // -- generate and store coefficient matrices.

  node_out() << "  generating: coefficient matrices..." << '\n';

  generate_all_coefficients<prec>(*pde);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.get_time_steps() < 1)
    return 0;

  node_out() << "--- begin time loop staging ---" << '\n';
  // -- allocate/setup for batch gemm

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

  host_workspace<prec> host_space(*pde, subgrid, default_workspace_cpu_MB);
  std::vector<element_chunk> const chunks = assign_elements(
      subgrid, get_num_chunks(plan.at(my_rank), *pde, default_workspace_MB));
  device_workspace<prec> dev_space(*pde, plan.at(my_rank), chunks);

  auto const get_MB = [&](int num_elems) {
    int64_t const bytes    = num_elems * sizeof(prec);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  node_out() << "allocating workspace..." << '\n';

  node_out() << "input vector size (MB): "
             << get_MB(dev_space.batch_input.size()) << '\n';
  node_out() << "kronmult output space size (MB): "
             << get_MB(dev_space.reduction_space.size()) << '\n';
  node_out() << "kronmult working space size (MB): "
             << get_MB(dev_space.batch_intermediate.size()) << '\n';
  node_out() << "output vector size (MB): "
             << get_MB(dev_space.batch_output.size()) << '\n';
  auto const &unit_vect = dev_space.get_unit_vector();
  node_out() << "reduction vector size (MB): " << get_MB(unit_vect.size())
             << '\n';

  node_out() << "explicit time loop workspace size (host) (MB): "
             << host_space.size_MB() << '\n';

  host_space.x = initial_condition;

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
  wavelet_to_realspace<prec>(*pde, initial_condition, table,
                             default_workspace_cpu_MB, tmp_workspace,
                             real_space);

  // initialize realspace output
  std::string const realspace_output_name = "asgard_realspace";
  auto output_dataset_real =
      initialize_output_file(real_space, "asgard_realspace");
#endif

  // -- time loop

  prec const dt = pde->get_dt() * opts.get_cfl();
  node_out() << "--- begin time loop w/ dt " << dt << " ---\n";
  for (int i = 0; i < opts.get_time_steps(); ++i)
  {
    prec const time = i * dt;

    if (opts.using_implicit())
    {
      bool const update_system = i == 0;
      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt, opts.get_selected_solver(),
                            update_system);
    }
    else
    {
      // FIXME fold initial sources into host space
      explicit_time_advance(*pde, table, initial_sources, host_space, dev_space,
                            chunks, plan, time, dt);
    }

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      prec const time_multiplier = pde->exact_time((i + 1) * dt);

      fk::vector<prec> const analytic_solution_t =
          analytic_solution * time_multiplier;
      fk::vector<prec> const diff = host_space.x - analytic_solution_t;
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
    if (opts.write_at_step(i))
    {
      update_output_file(output_dataset, host_space.x);
    }

    /* transform from wavelet space to real space */
    if (opts.transform_at_step(i))
    {
      wavelet_to_realspace<prec>(*pde, host_space.x, table,
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
  auto const final_result =
      gather_results(host_space.x, plan, my_rank, segment_size);

  finalize_distribution();

  return 0;
}
