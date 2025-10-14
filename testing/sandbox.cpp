#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using prec = asgard::default_precision;

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  prog_opts options(argc, argv);
  pde_domain<prec> domain(2);

  options.default_start_levels = {3, };
  options.default_degree = 1;

  pde_scheme<prec> pde(options, domain);

  auto g1 = pde.new_term_group();

  auto id0 = pde.register_moment({0, 1});
  auto id1 = pde.register_moment({1, 1});

  // std::cout << id0() << "    " << id1() << '\n';

  auto g2 = pde.new_term_group();

  auto id2 = pde.register_moment({1, 1});
  auto id3 = pde.register_moment({2, 1});

  ignore(g1);
  ignore(g2);
  ignore(id0);
  ignore(id1);
  ignore(id2);
  ignore(id3);

  pde.print_moments();

  return 0;
}
