
//-----------------------------------------------------------------------------
//
// This file generates the main() for the Catch2 tests.
//
// We compile it here separately to cut down on recompilation times for the main
// software components.
//
//-----------------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "tests_general.hpp"

asgard::options make_options(std::vector<std::string> const arguments)
{
  return asgard::options(make_parser(arguments));
}

asgard::parser make_parser(std::vector<std::string> const arguments)
{
  std::vector<const char *> argv;
  argv.push_back("asgard");
  for (const auto &arg : arguments)
  {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  return asgard::parser(argv.size() - 1, argv.data());
}
