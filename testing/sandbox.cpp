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

#include "kronmult.hpp"
#include "asgard_kronmult_tests.hpp"

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

template<typename T>
void test_kronmult_cpu_v22(int dimensions, int n, int num_y, int output_length,
                          int num_matrices)
{
  auto data =
      make_kronmult_data<T>(dimensions, n, num_y, output_length, num_matrices);

  asgard::fk::vector<T> vA(num_matrices * n * n);
  for (int k=0; k<num_matrices; k++)
  {
    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        vA(k * n * n + i * n + j) = data->matrices[k * n * n + j * n + i];
      }
    }
  }

  asgard::fk::vector<int> iA(num_matrices * dimensions);
  auto ip = data->pointer_map.begin();
  for (int i = 0; i < data->num_batch; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
      iA(i*dimensions + j) = n * n * (*ip++);
    ip++;
  }

  asgard::kronmult_matrix<T>
    kmat(dimensions, n, num_y, output_length,
         asgard::fk::vector<int, asgard::mem_type::const_view>(iA),
         asgard::fk::vector<T, asgard::mem_type::const_view>(vA));

  kmat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());

  for(auto &x : data->output_y) std::cout << x << "  ";
  std::cout << "\n";
  for(auto &x : data->reference_y) std::cout << x << "  ";
  std::cout << "\n";

  //test_almost_equal(data->output_y, data->reference_y, 100);
}

int main(int, char**)
{

  test_kronmult_cpu_v22<double>(1, 2, 1, 1, 4);



  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
