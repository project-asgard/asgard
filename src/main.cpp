#include "batch.hpp"
#include "build_info.hpp"
#include "coefficients.hpp"
#include "connectivity.hpp"
#include "element_table.hpp"
#include "mem_usage.hpp"
#include "pde.hpp"
#include "predict.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"

// continuity_1 2 2 -> 0.2755(MB)

using prec = double;
int main(int argc, char **argv)
{
  std::cout << "Branch: " << GIT_BRANCH << std::endl;
  std::cout << "Commit Summary: " << GIT_COMMIT_HASH << GIT_COMMIT_SUMMARY
            << std::endl;
  std::cout << "This executable was built on " << BUILD_TIME << std::endl;

  options opts(argc, argv);

  // -- parse user input and generate pde
  std::cout << "generating: pde..." << std::endl;
  auto pde = make_PDE<prec>(opts.get_selected_pde(), opts.get_level(),
                            opts.get_degree());

  // sync up options object in case pde defaults were loaded
  // assume uniform level and degree across dimensions
  opts.update_level(pde->get_dimensions()[0].get_level());
  opts.update_degree(pde->get_dimensions()[0].get_degree());

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
  auto const degree = pde->get_dimensions()[0].get_degree();

  std::cout << "ASGarD problem configuration:" << '\n';
  std::cout << "  selected PDE: " << opts.get_pde_string() << '\n';
  std::cout << "  level: " << opts.get_level() << '\n';
  std::cout << "  degree: " << opts.get_degree() << '\n';
  std::cout << "  N steps: " << opts.get_time_steps() << '\n';
  std::cout << "  write freq: " << opts.get_write_frequency() << '\n';
  std::cout << "  vis. freq: " << opts.get_visualization_frequency() << '\n';
  std::cout << "  implicit: " << opts.using_implicit() << '\n';
  std::cout << "  full grid: " << opts.using_full_grid() << '\n';
  std::cout << "  CFL number: " << opts.get_cfl() << '\n';
  std::cout << "  Poisson solve: " << opts.do_poisson_solve() << '\n';

  // -- print out time and memory estimates based on profiling
  std::pair<std::string, double> runtime_info = expected_time(
      opts.get_selected_pde(), opts.get_level(), opts.get_degree());
  std::cout << "Predicted compute time (seconds): " << runtime_info.second
            << std::endl;
  std::cout << runtime_info.first << std::endl;

  std::pair<std::string, double> mem_usage_info = total_mem_usage(
      opts.get_selected_pde(), opts.get_level(), opts.get_degree());
  std::cout << "Predicted total mem usage (MB): " << mem_usage_info.second
            << std::endl;
  std::cout << mem_usage_info.first << std::endl;

  std::cout << "--- begin setup ---" << std::endl;

  // -- create forward/reverse mapping between elements and indices
  std::cout << "  generating: element table..." << std::endl;
  element_table const table = element_table(opts, pde->num_dims);

  // -- generate initial condition vector.
  std::cout << "  generating: initial conditions..." << std::endl;
  fk::vector<prec> const initial_condition = [&pde, &table, degree]() {
    std::vector<fk::vector<prec>> initial_conditions;
    for (dimension<prec> const &dim : pde->get_dimensions())
    {
      initial_conditions.push_back(
          forward_transform<prec>(dim, dim.initial_condition));
    }
    return combine_dimensions(degree, table, initial_conditions);
  }();

  // -- generate source vectors.
  // these will be scaled later according to the simulation time applied
  // with their own time-scaling functions
  std::cout << "  generating: source vectors..." << std::endl;
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
  std::cout << "  generating: analytic solution at t=0 ..." << std::endl;
  fk::vector<prec> const analytic_solution = [&pde, &table, degree]() {
    std::vector<fk::vector<prec>> analytic_solutions_D;
    for (int d = 0; d < pde->num_dims; d++)
    {
      analytic_solutions_D.push_back(forward_transform<prec>(
          pde->get_dimensions()[d], pde->exact_vector_funcs[d]));
    }
    return combine_dimensions(degree, table, analytic_solutions_D);
  }();

  // -- generate and store coefficient matrices.
  std::cout << "  generating: coefficient matrices..." << std::endl;
  for (int i = 0; i < pde->num_dims; ++i)
  {
    dimension<prec> const dim = pde->get_dimensions()[i];
    for (int j = 0; j < pde->num_terms; ++j)
    {
      term<prec> const partial_term = pde->get_terms()[j][i];
      fk::matrix<prec> const coeff  = generate_coefficients(dim, partial_term);
      pde->set_coefficients(coeff, j, i);
    }
  }

  // this is to bail out for further profiling/development on the setup routines
  if (opts.get_time_steps() < 1)
    return 0;

  std::cout << "--- begin time loop staging ---" << std::endl;
  // -- allocate/setup for batch gemm
  auto const get_MB = [&](int num_elems) {
    uint64_t const bytes   = num_elems * sizeof(prec);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  int const default_workspace_MB = 1000;

  std::cout << "allocating workspace..." << std::endl;
  explicit_system<prec> system(*pde, table, default_workspace_MB);
  std::cout << "input vector size (MB): " << get_MB(system.x.size())
            << std::endl;
  std::cout << "kronmult output space size (MB): " << get_MB(system.y.size())
            << std::endl;
  std::cout << "kronmult working space size (MB): "
            << get_MB(system.work.size()) << std::endl;
  std::cout << "output vector size (MB): " << get_MB(system.fx.size())
            << std::endl;
  auto const &unit_vect = system.get_unit_vector();
  std::cout << "reduction vector size (MB): " << get_MB(unit_vect.size())
            << std::endl;

  explicit_workspace<prec> time_advance_work(system);
  std::cout << "explicit time loop workspace size (MB): "
            << get_MB(system.x.size() * 5) << std::endl;

  // call to build batches
  std::cout << "generating: batch lists..." << std::endl;
  auto const work_set =
      build_work_set(*pde, table, system, default_workspace_MB);

  std::cout << "allocating time loop working space, size (MB): "
            << get_MB(system.x.size() * 5) << std::endl;
  fk::vector<prec> scaled_source(system.x.size());
  fk::vector<prec> x_orig(system.x.size());
  std::vector<fk::vector<prec>> workspace(3, fk::vector<prec>(system.x.size()));

  // -- time loop
  std::cout << "--- begin time loop ---" << std::endl;
  prec const dt = pde->get_dt() * opts.get_cfl();
  for (int i = 0; i < opts.get_time_steps(); ++i)
  {
    prec const time = i * dt;

    explicit_time_advance(*pde, initial_sources, system, time_advance_work,
                          work_set, time, dt);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      prec const time_multiplier = pde->exact_time((i + 1) * dt);

      fk::vector<prec> const analytic_solution_t =
          analytic_solution * time_multiplier;
      fk::vector<prec> const diff =
          system.fx - (analytic_solution_t * time_multiplier);
      prec const RMSE = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        prec const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();
      auto const relative_error = RMSE / inf_norm(analytic_solution_t) * 100;
      std::cout << "RMSE (numeric-analytic) [wavelet]: " << RMSE << std::endl;
      std::cout << "Relative difference (numeric-analytic) [wavelet]: "
                << relative_error << " %" << std::endl;
    }

    std::cout << "timestep: " << i << " complete" << std::endl;
  }

  std::cout << "--- simulation complete ---" << std::endl;
  return 0;
}
