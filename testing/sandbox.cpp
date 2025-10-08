#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  int const level  = 2;
  int const degree = 1;

  pde_domain<prec> domain(2);
  hierarchy_manipulator<prec> hier(degree, domain);

  connection_patterns conn(level);

  quadmd_manager<double> quadmd(domain, hier, conn);


  return 0;
}
