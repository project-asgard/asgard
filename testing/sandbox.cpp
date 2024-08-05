#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

int main(int argc, char **argv)
{
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  prog_opts opts(argc, argv);

  std::cout << " read start levels: \n";
  for (auto l : opts.start_levels)
    std::cout << l << "  ";
  std::cout << '\n';

  return 0;
}
