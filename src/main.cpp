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
#include "predict.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

using prec = double;
int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

  node_out() << "Branch: " << GIT_BRANCH << '\n';
  node_out() << "Commit Summary: " << GIT_COMMIT_HASH << GIT_COMMIT_SUMMARY
             << '\n';
  node_out() << "This executable was built on " << BUILD_TIME << '\n';

  options opts(argc, argv);

  if (opts.using_implicit())
  {
    // distribution of implicit time advance not yet implemented
    assert(num_ranks == 1);
  }

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
  node_out() << "  vis. freq: " << opts.get_visualization_frequency() << '\n';
  node_out() << "  implicit: " << opts.using_implicit() << '\n';
  node_out() << "  full grid: " << opts.using_full_grid() << '\n';
  node_out() << "  CFL number: " << opts.get_cfl() << '\n';
  node_out() << "  Poisson solve: " << opts.do_poisson_solve() << '\n';

  // -- print out time and memory estimates based on profiling
  std::pair<std::string, double> runtime_info = expected_time(
      opts.get_selected_pde(), opts.get_level(), opts.get_degree());
  node_out() << "Predicted compute time (seconds): " << runtime_info.second
             << '\n';
  node_out() << runtime_info.first << '\n';

  std::pair<std::string, double> mem_usage_info = total_mem_usage(
      opts.get_selected_pde(), opts.get_level(), opts.get_degree());
  node_out() << "Predicted total mem usage (MB): " << mem_usage_info.second
             << '\n';
  node_out() << mem_usage_info.first << '\n';

  node_out() << "--- begin setup ---" << '\n';

  // -- create forward/reverse mapping between elements and indices
  node_out() << "  generating: element table..." << '\n';
  element_table const table = element_table(opts, pde->num_dims);

  // -- generate initial condition vector.
  node_out() << "  generating: initial conditions..." << '\n';
  fk::vector<prec> const initial_condition = [&pde, &table, degree]() {
    std::vector<fk::vector<prec>> initial_conditions;
    for (dimension<prec> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<prec>(dim, dim.initial_condition));
    }
    return combine_dimensions(degree, table, initial_conditions);
  }();

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE
  auto output_dataset = initialize_output_file(initial_condition);
#endif

  // -- generate source vectors.
  // these will be scaled later according to the simulation time applied
  // with their own time-scaling functions
  node_out() << "  generating: source vectors..." << '\n';
  std::vector<fk::vector<prec>> const initial_sources = [&pde, &table,
                                                         degree]() {
    std::vector<fk::vector<prec>> initial_sources;
    for (source<prec> const &source : pde->sources)
    {
      // gather contributions from each dim for this source, in wavelet space
      std::vector<fk::vector<prec>> initial_sources_dim;
      for (int i = 0; i < pde->num_dims; ++i)
      {
        initial_sources_dim.push_back(forward_transform<prec>(
            pde->get_dimensions()[i], source.source_funcs[i]));
      }
      // combine those contributions to form the unscaled source vector
      initial_sources.push_back(
          combine_dimensions(degree, table, initial_sources_dim));
    }
    return initial_sources;
  }();

  // -- generate analytic solution vector.
  node_out() << "  generating: analytic solution at t=0 ..." << '\n';
  fk::vector<prec> const analytic_solution = [&pde, &table, degree]() {
    std::vector<fk::vector<prec>> analytic_solutions_D;
    for (int d = 0; d < pde->num_dims; ++d)
    {
      analytic_solutions_D.push_back(forward_transform<prec>(
          pde->get_dimensions()[d], pde->exact_vector_funcs[d]));
    }
    return combine_dimensions(degree, table, analytic_solutions_D);
  }();

  // -- generate and store coefficient matrices.

  std::cout << "  generating: coefficient matrices..." << '\n';

  generate_all_coefficients<prec>(*pde);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.get_time_steps() < 1)
    return 0;

  node_out() << "--- begin time loop staging ---" << '\n';
  // -- allocate/setup for batch gemm
  // FIXME these comments need update
  // Our default workspace size is ~1GB.

  // This 1GB doesn't include coefficient matrices, element table,
  // or time advance workspace - only the primary memory consumers (kronmult
  // intermediate and result workspaces).
  //
  // FIXME eventually going to be settable from the cmake
  static int const default_workspace_MB = 7000;

  auto const plan = get_plan(num_ranks, table);
  host_workspace<prec> host_space(*pde, table);
  std::vector<element_chunk> const chunks = assign_elements(
      table, get_num_chunks(plan.at(my_rank), *pde, default_workspace_MB));
  rank_workspace<prec> rank_space(*pde, chunks);

  auto const get_MB = [&](int num_elems) {
    uint64_t const bytes   = num_elems * sizeof(prec);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  node_out() << "allocating workspace..." << '\n';

  node_out() << "input vector size (MB): "
             << get_MB(rank_space.batch_input.size()) << '\n';
  node_out() << "kronmult output space size (MB): "
             << get_MB(rank_space.reduction_space.size()) << '\n';
  node_out() << "kronmult working space size (MB): "
             << get_MB(rank_space.batch_intermediate.size()) << '\n';
  node_out() << "output vector size (MB): "
             << get_MB(rank_space.batch_output.size()) << '\n';
  auto const &unit_vect = rank_space.get_unit_vector();
  node_out() << "reduction vector size (MB): " << get_MB(unit_vect.size())
             << '\n';

  node_out() << "explicit time loop workspace size (host) (MB): "
             << host_space.size_MB() << '\n';

  host_space.x = initial_condition;

  // -- time loop
  node_out() << "--- begin time loop ---" << '\n';
  prec const dt = pde->get_dt() * opts.get_cfl();
  for (int i = 0; i < opts.get_time_steps(); ++i)
  {
    prec const time = i * dt;

    if (opts.using_implicit())
    {
      bool update_system = i == 0;
      implicit_time_advance(*pde, table, initial_sources, host_space, chunks,
                            time, dt, update_system);
    }
    else
    {
      // FIXME fold initial sources into host space
      explicit_time_advance(*pde, table, initial_sources, host_space,
                            rank_space, chunks, plan, my_rank, time, dt);
    }

    // FIXME this needs to be wrapped and serialized
    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      prec const time_multiplier = pde->exact_time((i + 1) * dt);

      fk::vector<prec> const analytic_solution_t =
          analytic_solution * time_multiplier;
      fk::vector<prec> const diff = host_space.fx - analytic_solution_t;
      prec const RMSE             = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        prec const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();

      auto const relative_error = RMSE / inf_norm(analytic_solution_t) * 100;
      node_out() << "RMSE (numeric-analytic) [wavelet]: " << RMSE << '\n';
      node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                 << relative_error << " %" << '\n';
    }

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
    update_output_file(output_dataset, host_space.fx);
#endif

    node_out() << "timestep: " << i << " complete" << '\n';
  }

  node_out() << "--- simulation complete ---" << '\n';

  finalize_distribution();

  return 0;
}
