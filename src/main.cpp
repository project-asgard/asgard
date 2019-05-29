#include "batch.hpp"
#include "coefficients.hpp"
#include "connectivity.hpp"
#include "element_table.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"

using prec = double;
int main(int argc, char **argv)
{
  options opts(argc, argv);

  std::cout << "--- begin setup ---" << std::endl;

  // -- parse user input and generate pde
  std::cout << "generating: pde..." << std::endl;
  auto pde = make_PDE<prec>(opts.get_selected_pde(), opts.get_level(),
                            opts.get_degree());

  // sync up options object in case pde defaults were loaded
  // assume uniform level and degree across dimensions
  opts.update_level(pde->get_dimensions()[0].get_level());
  opts.update_degree(pde->get_dimensions()[0].get_degree());

  // -- create forward/reverse mapping between elements and indices
  std::cout << "generating: element table..." << std::endl;
  element_table const table = element_table(opts, pde->num_dims);

  // -- generate initial condition vector.
  std::cout << "generating: initial conditions..." << std::endl;
  std::vector<fk::vector<prec>> initial_conditions;
  for (dimension<prec> const &dim : pde->get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<prec>(dim, dim.initial_condition));
  }
  fk::vector<prec> const initial_condition =
      combine_dimensions(pde->get_dimensions()[0], table, initial_conditions);

  // -- generate source vectors.
  // these will be scaled later according to the simulation time applied with
  // their own time-scaling functions
  std::cout << "generating: source vectors..." << std::endl;
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
    initial_sources.push_back(combine_dimensions(pde->get_dimensions()[0],
                                                 table, initial_sources_dim));
  }

  // -- generate and store coefficient matrices.
  std::cout << "generating: coefficient matrices..." << std::endl;
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
  auto get_MB = [&](int num_elems) {
    uint64_t bytes   = num_elems * sizeof(prec);
    double megabytes = bytes * 1e-6;
    return megabytes;
  };

  // input vector x
  int const elem_size =
      std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
  std::cout << "allocating input vector, size (MB): "
            << get_MB(table.size() * elem_size) << std::endl;
  fk::vector<prec> x(table.size() * elem_size);
  x = initial_condition;

  // intermediate output spaces for batched gemm
  std::cout << "allocating kronmult output space, size (MB): "
            << get_MB(x.size() * table.size() * pde->num_terms) << std::endl;
  fk::vector<prec> y(x.size() * table.size() * pde->num_terms);

  std::cout << "allocating kronmult working space, size (MB): "
            << get_MB(x.size() * table.size() * pde->num_terms *
                      (pde->num_dims - 1))
            << std::endl;
  fk::vector<prec> work(x.size() * table.size() * pde->num_terms *
                        (pde->num_dims - 1));

  // output vector fx
  std::cout << "allocating output vector, size (MB): " << get_MB(x.size())
            << std::endl;
  fk::vector<prec> fx(x.size());

  // setup reduction vector
  int const items_to_reduce = pde->num_terms * table.size();
  std::cout << "allocating reduction vector, size (MB): "
            << get_MB(items_to_reduce) << std::endl;
  fk::vector<prec> const unit_vector = [&] {
    fk::vector<prec> builder(items_to_reduce);
    std::fill(builder.begin(), builder.end(), 1.0);
    return builder;
  }();

  // call to build batches
  std::cout << "generating: batch lists..." << std::endl;
  std::vector<batch_operands_set<prec>> const batches =
      build_batches(*pde, table, x, y, work, unit_vector, fx);

  // these vectors used for intermediate results in time advance
  std::cout << "allocating time loop working space, size (MB): "
            << get_MB(x.size() * 5) << std::endl;
  fk::vector<prec> scaled_source(x.size());
  fk::vector<prec> x_orig(x.size());
  std::vector<fk::vector<prec>> workspace(3, fk::vector<prec>(x.size()));

  // -- time loop
  std::cout << "--- begin time loop ---" << std::endl;
  prec const dt = pde->get_dt() * opts.get_cfl();
  for (int i = 0; i < opts.get_time_steps(); ++i)
  {
    prec const time = i * dt;
    explicit_time_advance(*pde, x, x_orig, fx, scaled_source, initial_sources,
                          workspace, batches, time, dt);

    std::cout << "timestep: " << i << " complete" << std::endl;
  }

  std::cout << "--- simulation complete ---" << std::endl;
  return 0;
}
