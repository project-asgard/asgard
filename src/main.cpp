#include "element_table.hpp"
#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

int main(int argc, char **argv)
{
  options opts(argc, argv);
  auto pde        = make_PDE<double>(opts.get_selected_pde(), opts.get_level(),
                              opts.get_degree());
  element_table e = element_table(opts, pde->num_dims);
  return 0;
}
