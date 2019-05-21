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
  // -- parse user input and generate pde
  options opts(argc, argv);
  auto pde = make_PDE<prec>(opts.get_selected_pde(), opts.get_level(),
                            opts.get_degree());
  
  // sync up options object in case pde defaults were loaded
  // assume uniform level and degree across dimensions
  opts.update_level(pde->get_dimensions()[0].get_level());
  opts.update_degree(pde->get_dimensions()[0].get_degree());

  // -- create forward/reverse mapping between elements and indices
  element_table const table = element_table(opts, pde->num_dims);

  // -- generate initial condition vector.
  prec const initial_scale = 1.0;
  std::vector<fk::vector<prec>> initial_conditions;
  for (dimension<prec> const &dim : pde->get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<prec>(dim, dim.initial_condition));
  }
  fk::vector<prec> const initial_condition = combine_dimensions(
      pde->get_dimensions()[0], table, initial_conditions, initial_scale);

  // -- generate and store coefficient matrices.
  prec const initial_time = 0.0;
  for (int i = 0; i < pde->num_dims; ++i)
  {
    dimension<prec> const dim = pde->get_dimensions()[i];
    for (int j = 0; j < pde->num_terms; ++j)
    {
      term<prec> const partial_term = pde->get_terms()[j][i];
      fk::matrix<prec> const coeff =
          generate_coefficients(dim, partial_term, initial_time);
      pde->set_coefficients(coeff, j, i);
    }
  }

  // -- allocate/setup for batch gemm

  // input vector x
  int const elem_size =
      std::pow(pde->get_dimensions()[0].get_degree(), pde->num_dims);
  fk::vector<prec> x(table.size() * elem_size);
  x = initial_condition;

  // intermediate output spaces for batched gemm
  fk::vector<prec> y(x.size() * table.size() * pde->num_terms);
  fk::vector<prec> work(elem_size * table.size() * table.size() *
                        pde->num_terms * (pde->num_dims - 1));

  // output vector fx
  fk::vector<prec> fx(x.size());

  // setup reduction vector
  int const items_to_reduce          = pde->num_terms * table.size();
  fk::vector<prec> const unit_vector = [&] {
    fk::vector<prec> builder(items_to_reduce);
    std::fill(builder.begin(), builder.end(), 1.0);
    return builder;
  }();

  // call to build batches
  std::vector<batch_operands_set<prec>> const batches =
      build_batches(*pde, table, x, y, work, unit_vector, fx);

  // -- generate sources.
  // these will be scaled later for time
  std::vector<fk::vector<prec>> initial_sources;

  for (source<prec> const &source : pde->sources)
  {
    std::vector<fk::vector<prec>> initial_sources_dim;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      initial_sources_dim.push_back(forward_transform<prec>(
          pde->get_dimensions()[i], source.source_funcs[i]));
    }

    initial_sources.push_back(combine_dimensions(
        pde->get_dimensions()[0], table, initial_sources_dim, initial_scale));
  }

  // these vectors used for intermediate results in time advance
  fk::vector<prec> scaled_source(x.size());
  fk::vector<prec> x_orig(x.size());
  std::vector<fk::vector<prec>> workspace(3, fk::vector<prec>(x.size()));

  // -- time loop
  prec const dt = pde->get_dt() * opts.get_cfl();
  for (int i = 0; i < opts.get_time_steps(); ++i)
  {
    prec const time = i * dt;
    explicit_time_advance(*pde, x, x_orig, fx, scaled_source, initial_sources,
                          workspace, batches, time, dt);
  }

  return 0;
}
