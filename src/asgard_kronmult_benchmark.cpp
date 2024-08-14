#include "asgard_kronmult_tests.hpp"

using precision = asgard::default_precision;

int main(int argc, char **argv)
{
  std::cout << " currently the benchmark does not work \n";
  return 0;

  if (argc < 6)
  {
    std::cout
#ifdef ASGARD_ENABLE_DOUBLE
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

  asgard::local_kronmult_matrix<precision> mat;

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
#ifdef ASGARD_ENABLE_DOUBLE
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
