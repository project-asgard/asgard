#include "asgard.hpp"

using precision = asgard::default_precision;

int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = asgard::initialize_distribution();

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    asgard::finalize_distribution();
    return 0;
  }

  // custom projects may implement their own inputs
  //   in addition to ASGarD standard ones
  // however, this the main executable and all
  //   unknown cli commands are to be treated as errors
  // first we generate warnigns for the user
  bool constexpr ignore_unknown = false;
  // -- parse cli
  asgard::prog_opts const cli_input(argc, argv, ignore_unknown);

  // if there were unknown options, throw an error
  if (not cli_input.externals.empty())
    throw std::runtime_error("encountered unrecognized command line option(s)");

  // main call to asgard, does all the work
  asgard::simulate_builtin<precision>(cli_input);

  asgard::finalize_distribution();

  return 0;
}
