#include "asgard.hpp"

using precision = asgard::default_precision;

int main(int argc, char **argv)
{
  asgard::prog_opts const options(argc, argv);

  // if there were unknown options, throw an error
  options.throw_if_invalid();

  if (options.show_help) {
    options.print_help();
    return 0;
  }

  if (options.show_version) {
    options.print_version_help();
    return 0;
  }

  std::cerr << "the 'asgard' utility can only print version and options help\n";
  std::cerr << "looking for the pde files, check the other executables, e.g., continuity or elliptic\n";

  return 1;
}
