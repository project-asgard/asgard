#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

#include "asgard_kronmult_tests.hpp"

using namespace asgard;

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

template<typename T>
void test_almost_equal(std::vector<T> const &x, std::vector<T> const &y,
                       int = 10)
{
  T err = 0.0;
  for(size_t i=0; i<x.size(); i++)
    err = std::max(std::abs(x[i] - y[i]), err);

  std::cout << " error: " << err << "\n";
}

//template<typename P>
//void test_kronmult_welem(int dimensions, int n, int num_terms,
//                         int num_1d_blocks)
//{
//  constexpr bool precompute = true;
//
//  auto data = make_kronmult_welem<P, precompute>(dimensions, n, num_terms, num_1d_blocks);
//
//  cpu_dense<P>(dimensions, n, data->num_rows(), data->num_rows(), num_terms,
//               data->elem.data(), 0, 0, data->get_offsets().data(), num_1d_blocks,
//               P{1.0}, data->input_x.data(), P{1.0}, data->output_y.data());
//
//  test_almost_equal(data->output_y, data->reference_y, 100);
//}


int main(int, char **)
{
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  //test_kronmult_welem<double>(2, 1, 3, 7);
  //test_kronmult_welem<double>(2, 2, 3, 7);
  //test_kronmult_welem<double>(2, 3, 3, 7);
  //test_kronmult_welem<double>(2, 4, 3, 7);

  return 0;
}
