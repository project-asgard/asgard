#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

using P = asgard::default_precision;

int main(int argc, char **argv)
{
  std::ignore = argc;
  std::ignore = argv;

  int const degree = 2;
  int const pdof   = degree + 1;

  auto diff = legendre::poly2diff(degree);

  std::cout << " ------ res --------- \n";
  for (int i = 0; i < pdof; i++) {
    for (int j = 0; j < pdof; j++)
      std::cout << diff[i][j] << "    ";
    std::cout << '\n';
  }

  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
