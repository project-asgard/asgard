
#include <chrono>
#include <string>

#include "asgard_kronmult_tests.hpp"

#ifdef ASGARD_USE_DOUBLE_PREC
using precision = double;
#else
using precision = float;
#endif

int main(int argc, char **argv)
{
  if (argc < 6)
  {
    std::cout
#ifdef ASGARD_USE_DOUBLE_PREC
        << "\n build for double precision"
#else
        << "\n build for single precision"
#endif
#ifdef ASGARD_USE_CUDA
        << " using CUDA backend\n"
#else
        << " using CPU backend\n"
#endif
        << "\n Usage:\n"
        << "./asgard_kronmult_benchmark <dimensions> <n> <num_rows> "
           "<num_terms> <num_matrices>\n\n"
        << " e.g., ./asgard_kronmult_benchmark 3 2 688  3 36864\n"
        << "       ./asgard_kronmult_benchmark 3 2 4096 3 589824\n"
        << "       ./asgard_kronmult_benchmark 2 2 256  2 16384\n"
        << "       ./asgard_kronmult_benchmark 2 2 1280 2 262144\n"
        << "\n see the documentation in asgard_kronmult_tests.hpp template "
           "make_kronmult_data()\n\n";
    return 1;
  }

  int dimensions   = std::stoi(argv[1]);
  int n            = std::stoi(argv[2]);
  int num_rows     = std::stoi(argv[3]);
  int num_terms    = std::stoi(argv[4]);
  int num_matrices = std::stoi(argv[5]);

  std::cout << "benchmarking:\n"
            << "dimensions: " << dimensions << "  n: " << n
            << "  num_rows: " << num_rows << "  num_terms: " << num_terms
            << "  num_matrices: " << num_matrices << "\n";

  constexpr bool compute_refrence_solution = false;

  int const num_batch = num_rows * num_rows * num_terms;
  auto time_start     = std::chrono::system_clock::now();

  auto data = make_kronmult_data<precision, compute_refrence_solution>(
      dimensions, n, num_rows, num_terms, num_matrices);

  auto time_end = std::chrono::system_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(time_end - time_start).count();

  std::cout << "benchmark setup time: " << elapsed / 1000 << " seconds.\n";

  double flops = 1; // compute 2 * d * n^(dimensions+1) * num_batch
  for (int i = 0; i < dimensions + 1; i++)
    flops *= n;
  flops *= 2.0 * dimensions * num_batch;
  // Gflops is flops * 1.E-9, milliseconds is seconds * 1.E+3
  double unit_scale       = 1.E-6;
  constexpr int num_tests = 100;

  asgard::fk::vector<precision> vA(num_matrices * n * n);
  for (int k = 0; k < num_matrices * n * n; k++)
    vA(k) = data->matrices[k];

  asgard::fk::vector<int> iA(num_batch * dimensions);
  auto ip = data->pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
    {
      iA(i * dimensions + j) = n * n * (*ip);
    }
    ip++;
  }

  asgard::fk::vector<int> row_indx(num_rows * num_rows);
  asgard::fk::vector<int> col_indx(num_rows * num_rows);

  asgard::kronmult_matrix<precision> mat;

#ifdef ASGARD_USE_CUDA
  int tensor_size = n;
  for(int d=1; d<dimensions; d++)
    tensor_size *= n;

  for(int i=0; i<num_rows; i++)
  {
    for(int j=0; j<num_rows; j++)
    {
      row_indx[i * num_rows + j] = i * tensor_size;
      col_indx[i * num_rows + j] = j * tensor_size;
    }
  }

  mat = asgard::kronmult_matrix<precision>(
      dimensions, n, num_rows, num_rows, num_terms,
      row_indx.clone_onto_device(),
      col_indx.clone_onto_device(),
      iA.clone_onto_device(),
      vA.clone_onto_device()
  );
#else
  for(int i=0; i<num_rows; i++)
  {
    row_indx[i] = i * num_rows;
    for(int j=0; j<num_rows; j++)
      col_indx[row_indx[i] + j]= j;
  }
  row_indx[num_rows] = col_indx.size();

  mat = asgard::kronmult_matrix<precision>(
      dimensions, n, num_rows, num_rows, num_terms, std::move(row_indx),
      std::move(col_indx), std::move(iA), std::move(vA));
#endif

  // dry run to wake up the devices
  mat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());

  time_start = std::chrono::system_clock::now();
  for (int i = 0; i < num_tests; i++)
  {
    mat.apply(1.0, data->input_x.data(), 1.0, data->output_y.data());
  }
  time_end = std::chrono::system_clock::now();
  double felapsed =
      std::chrono::duration<double, std::milli>(time_end - time_start).count();

  std::cout << std::fixed << std::setprecision(4);
#ifdef ASGARD_USE_DOUBLE_PREC
  std::cout << "double precision: ";
#else
  std::cout << "single precision: ";
#endif
  if (felapsed == 0)
  {
    std::cout
        << " test finished too fast to be accurately timed, use larger sizes\n";
  }
  else
  {
    std::cout << unit_scale * (num_tests * flops / felapsed)
              << " Gflops / second.\n";
  }

  return 0;
}
