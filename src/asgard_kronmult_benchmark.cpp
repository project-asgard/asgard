
#include <chrono>
#include <string>

#include "asgard_kronmult_tests.hpp"

int main(int argc, char **argv)
{
  if (argc < 6)
  {
    std::cout
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

  auto fdata = make_kronmult_data<float, compute_refrence_solution>(
      dimensions, n, num_rows, num_terms, num_matrices);
  auto ddata = make_kronmult_data<double, compute_refrence_solution>(
      dimensions, n, num_rows, num_terms, num_matrices);

  auto time_end = std::chrono::system_clock::now();
  double elapsed =
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_end - time_start)
                              .count());

  std::cout << "benchmark setup time: " << elapsed / 1000 << " seconds.\n";

  double flops = 1; // compute 2 * d * n^(dimensions+1) * num_batch
  for (int i = 0; i < dimensions + 1; i++)
    flops *= n;
  flops *= 2.0 * dimensions * num_batch;
  // Gflops is flops * 1.E-9, milliseconds is seconds * 1.E+3
  double unit_scale       = 1.E-6;
  constexpr int num_tests = 100;

  asgard::fk::vector<float> fvA(num_matrices * n * n);
  asgard::fk::vector<double> dvA(num_matrices * n * n);
  for (int k = 0; k < num_matrices * n * n; k++)
  {
    fvA(k) = fdata->matrices[k];
    dvA(k) = ddata->matrices[k];
  }

  asgard::fk::vector<int> fiA(num_batch * dimensions);
  asgard::fk::vector<int> diA(num_batch * dimensions);
  auto ip = fdata->pointer_map.begin();
  for (int i = 0; i < num_batch; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
    {
      fiA(i * dimensions + j) = n * n * (*ip);
      diA(i * dimensions + j) = n * n * (*ip++);
    }
    ip++;
  }

  asgard::kronmult_matrix<float> fmat(
      dimensions, n, num_rows, num_rows, num_terms,
      asgard::fk::vector<int, asgard::mem_type::const_view>(fiA),
      asgard::fk::vector<float, asgard::mem_type::const_view>(fvA));
  asgard::kronmult_matrix<double> dmat(
      dimensions, n, num_rows, num_rows, num_terms,
      asgard::fk::vector<int, asgard::mem_type::const_view>(diA),
      asgard::fk::vector<double, asgard::mem_type::const_view>(dvA));

  // dry run to wake up the devices
  fmat.apply(1.0, fdata->input_x.data(), 1.0, fdata->output_y.data());

  time_start = std::chrono::system_clock::now();
  for (int i = 0; i < num_tests; i++)
  {
    fmat.apply(1.0, fdata->input_x.data(), 1.0, fdata->output_y.data());
  }
  time_end = std::chrono::system_clock::now();
  double felapsed =
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_end - time_start)
                              .count());

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "single precision: ";
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

  dmat.apply(1.0, ddata->input_x.data(), 1.0, ddata->output_y.data());

  time_start = std::chrono::system_clock::now();
  for (int i = 0; i < num_tests; i++)
  {
    dmat.apply(1.0, ddata->input_x.data(), 1.0, ddata->output_y.data());
  }
  time_end = std::chrono::system_clock::now();
  double delapsed =
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_end - time_start)
                              .count());

  std::cout << "double precision: ";
  if (delapsed == 0)
  {
    std::cout
        << " test finished too fast to be accurately timed, use larger sizes\n";
  }
  else
  {
    std::cout << unit_scale * (num_tests * flops / delapsed)
              << " Gflops / second.\n";
  }

  return 0;
}
