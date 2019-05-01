#include "coefficients.hpp"
#include "connectivity.hpp"
#include "element_table.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "transformations.hpp"

int main(int argc, char **argv)
{
  options opts(argc, argv);
  auto pde = make_PDE<double>(opts.get_selected_pde(), opts.get_level(),
                              opts.get_degree());

  element_table table = element_table(opts, pde->num_dims);

  // list_set connectivity = make_connectivity(table, pde->num_dims,
  //                                          opts.get_level(),
  //                                          opts.get_level());

  // this should probably be wrapped in a function, doing it inline for now
  std::vector<fk::vector<double>> initial_conditions;
  for (dimension<double> const dim : pde->get_dimensions())
  {
    initial_conditions.push_back(
        forward_transform<double>(dim, dim.initial_condition));
  }
  double const start = 1.0;
  // the combine dimensions function will have to be modified for variable
  // deg/lev among dimensions
  fk::vector<double> initial_condition = combine_dimensions(
      pde->get_dimensions()[0], table, initial_conditions, start);

  // same for this, wrapped up in a function
  // store sources, will scale later for time
  std::vector<fk::vector<double>> initial_sources;
  for (source<double> const source : pde->sources)
  {
    std::vector<fk::vector<double>> initial_sources_dim;
    for (int i = 0; i < pde->num_dims; ++i)
    {
      initial_sources_dim.push_back(forward_transform<double>(
          pde->get_dimensions()[i], source.source_funcs[i]));
    }

    initial_sources.push_back(combine_dimensions(
        pde->get_dimensions()[0], table, initial_sources_dim, start));
  }

  // make the terms. also another function. also, should time be one here, or
  // zero...
  double const init_time = 0.0;
  for (std::vector<term<double>> const terms : pde->get_terms())
  {
    for (term<double> term : terms)
    {
      for (int i = 0; i < pde->num_dims; ++i)
      {
        dimension<double> const dim = pde->get_dimensions()[i];
        term.set_coefficients(dim, generate_coefficients(dim, term, init_time));
      }
    }
  }

  return 0;
}
