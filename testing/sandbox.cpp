#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"

#include "asgard_kronmult_tests.hpp"

using namespace asgard;

using prec = asgard::default_precision;

int main(int, char **)
{
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
