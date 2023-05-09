
#include <chrono>
#include <string>

#include "asgard_kronmult_tests.hpp"

int main(int argc, char **argv)
{
  if (argc < 6)
  {
    std::cout
        << "\n Usage:\n"
        << "./asgard_kronmult_benchmark <dimensions> <n> <num_y> "
           "<operator_length> <num_matrices>\n"
        << "\n e.g., ./asgard-kronmult-benchmark 2 2 4096 12288 1000\n"
        << "\n see the documentation in asgard_kronmult_tests.hpp template "
           "make_kronmult_data()\n\n";
    return 1;
  }

  int dimensions    = std::stoi(argv[1]);
  int n             = std::stoi(argv[2]);
  int num_y         = std::stoi(argv[3]);
  int operator_size = std::stoi(argv[4]);
  int num_matrices  = std::stoi(argv[5]);

  std::cout << "benchmarking:\n"
            << "dimensions: " << dimensions << "  n: " << n
            << "  num_y: " << num_y << "  operator_size: " << operator_size
            << "  num_matrices: " << num_matrices << "\n";

  constexpr bool compute_refrence_solution = false;
  auto time_start                          = std::chrono::system_clock::now();
  auto fdata = make_kronmult_data<float, compute_refrence_solution, false>(
      dimensions, n, num_y, operator_size, num_matrices);
  auto ddata = make_kronmult_data<double, compute_refrence_solution, false>(
      dimensions, n, num_y, operator_size, num_matrices);
  auto time_end = std::chrono::system_clock::now();
  double elapsed =
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_end - time_start)
                              .count());

  std::cout << "benchmark setup time: " << elapsed / 1000 << " seconds.\n";

  double flops = 1;
  for (int i = 0; i < dimensions + 1; i++)
    flops *= n;
  flops *= 2.0 * dimensions * num_y * operator_size;
  double unit_scale =
      1.E-6; // Gflops is flops * 1.E-9, milliseconds is seconds * 1.E+3
  constexpr int num_tests = 100;

  asgard::fk::vector<float> fvA(num_matrices * n * n);
  asgard::fk::vector<double> dvA(num_matrices * n * n);
  for (int k=0; k<num_matrices; k++)
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++){
        fvA(k * n * n + i * n + j) = fdata->matrices[k * n * n + j * n + i];
        dvA(k * n * n + i * n + j) = ddata->matrices[k * n * n + j * n + i];
      }

  asgard::fk::vector<int> fiA(fdata->num_batch * dimensions);
  asgard::fk::vector<int> diA(ddata->num_batch * dimensions);
  auto ip = fdata->pointer_map.begin();
  for (int i = 0; i < fdata->num_batch; i++)
  {
    ip++;
    for (int j = 0; j < dimensions; j++)
    {
      fiA(i*dimensions + j) = n * n * (*ip);
      diA(i*dimensions + j) = n * n * (*ip++);
    }
    ip++;
  }

  asgard::kronmult_matrix<float, asgard::resource::host>
    fmat(dimensions, n, num_y, operator_size,
         asgard::fk::vector<int, asgard::mem_type::const_view>(fiA),
         asgard::fk::vector<float, asgard::mem_type::const_view>(fvA));
  asgard::kronmult_matrix<double, asgard::resource::host>
    dmat(dimensions, n, num_y, operator_size,
         asgard::fk::vector<int, asgard::mem_type::const_view>(diA),
         asgard::fk::vector<double, asgard::mem_type::const_view>(dvA));

// dry run to wake up the devices
#ifdef ASGARD_USE_CUDA
  execute_gpu(dimensions, n, fdata->gpupA.data(), n, fdata->gpupX.data(),
              fdata->gpupY.data(), fdata->num_batch, fdata->output_size);
  cudaDeviceSynchronize();
#else
  //execute_cpu(dimensions, n, fdata->pA.data(), n, fdata->pX.data(),
  //            fdata->pY.data(), fdata->num_batch, fdata->output_size);
  fmat.apply(1.0, fdata->input_x.data(), 1.0, fdata->output_y.data());
#endif

  time_start = std::chrono::system_clock::now();
  for (int i = 0; i < num_tests; i++)
  {
#ifdef ASGARD_USE_CUDA
    execute_gpu(dimensions, n, fdata->gpupA.data(), n, fdata->gpupX.data(),
                fdata->gpupY.data(), fdata->num_batch, fdata->output_size);
    cudaDeviceSynchronize();
#else
    //execute_cpu(dimensions, n, fdata->pA.data(), n, fdata->pX.data(),
    //            fdata->pY.data(), fdata->num_batch, fdata->output_size);
    fmat.apply(1.0, fdata->input_x.data(), 1.0, fdata->output_y.data());
#endif
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

#ifdef ASGARD_USE_CUDA
  execute_gpu(dimensions, n, ddata->gpupA.data(), n, ddata->gpupX.data(),
              ddata->gpupY.data(), ddata->num_batch, ddata->output_size);
  cudaDeviceSynchronize();
#else
  //execute_cpu(dimensions, n, ddata->pA.data(), n, ddata->pX.data(),
  //            ddata->pY.data(), ddata->num_batch, ddata->output_size);
  dmat.apply(1.0, ddata->input_x.data(), 1.0, ddata->output_y.data());
#endif

  time_start = std::chrono::system_clock::now();
  for (int i = 0; i < num_tests; i++)
  {
#ifdef ASGARD_USE_CUDA
    execute_gpu(dimensions, n, ddata->gpupA.data(), n, ddata->gpupX.data(),
                ddata->gpupY.data(), ddata->num_batch, ddata->output_size);
    cudaDeviceSynchronize();
#else
    //execute_cpu(dimensions, n, ddata->pA.data(), n, ddata->pX.data(),
    //            ddata->pY.data(), ddata->num_batch, ddata->output_size);
    dmat.apply(1.0, ddata->input_x.data(), 1.0, ddata->output_y.data());
#endif
  }
  time_end = std::chrono::system_clock::now();
  double delapsed =
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_end - time_start)
                              .count());

  std::cout << "double precision: ";
  if (felapsed == 0)
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
