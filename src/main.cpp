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

  // -- parse cli
  asgard::parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    asgard::node_out() << "invalid cli string; exiting\n";
    exit(-1);
  }

  // main call to asgard, does all the work
  asgard::simulate<precision>(cli_input);

  asgard::finalize_distribution();

  return 0;
}
