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

  int const degree = 0;
  int const level  = 2;

  prog_opts options(argc, argv);
  options.default_start_levels = {3, };
  options.default_degree = degree;

  pde_domain<prec> domain(position_dims{1}, velocity_dims{2}, {{0, 1}, {-1, 1}, {0, 1}});

  hierarchy_manipulator<prec> hier(degree, domain);
  moments1d<prec> mom1d(2, degree, level, domain);

  moments_list mlist;
  mlist.add_moment({0, 0});
  mlist.add_moment({1, 0});
  mlist.add_moment({0, 1});
  mlist.add_moment({2, 2});
  moment_manager<prec> momd(domain, degree, std::move(mlist));



  return 0;
}
