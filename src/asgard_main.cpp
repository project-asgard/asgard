#include "asgard.hpp"

#include "asgard_cmakelog.hpp"

int main(int argc, char **argv)
{
  asgard::prog_opts const options(argc, argv);

  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f^2 = nu * f_xx + s(t, x)\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< the asgard executable accepts one more option >>\n";
    std::cout << "-cmake                              show a CMake build log\n\n";
    return 0;
  }

  // if there were unknown options, throw an error
  options.throw_if_argv_not_in({"-cmake", }, {});

  if (options.has_cli_entry("-cmake")) {
    show_cmake_log();
    return 0;
  }

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
  std::cerr << "those are either in the build folder or in <prefix>/share/asgard/pde/\n";

  return 1;
}
