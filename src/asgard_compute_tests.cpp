#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

#ifdef ASGARD_USE_GPU
template<typename vec_type_1, typename vec_type_2>
bool data_match(vec_type_1 const &v1, vec_type_2 const &v2)
{
  if (static_cast<int64_t>(v1.size()) != static_cast<int64_t>(v2.size()))
    return false;
  static_assert(std::is_same_v<typename vec_type_1::value_type, typename vec_type_2::value_type>);
  std::vector<typename vec_type_1::value_type> x = v1;
  std::vector<typename vec_type_2::value_type> y = v2;
  if (x.size() != y.size()) // something happened during copy
    return false;
  for (size_t i = 0; i < x.size(); i++)
    if (x[i] != y[i]) // this checks data copies, so it's OK for floating point numbers
      return false;
  return true;
}

template<typename TestType>
void gpu_vector_tests() {
  current_test<TestType> name_("gpu-vector");
  {
    gpu::vector<TestType> gpu0; // make empty
    tassert(gpu0.size() == 0);
    tassert(gpu0.data() == nullptr);
    tassert(gpu0.empty());

    gpu0.resize(10); // resize
    tassert(gpu0.size() == 10);
    tassert(gpu0.data() != nullptr);
    tassert(not gpu0.empty());

    gpu0 = gpu::vector<TestType>(); // move-assign
    tassert(gpu0.size() == 0);
    tassert(gpu0.data() == nullptr);

    std::vector<TestType> cpu1 = {1, 2, 3, 4};
    gpu::vector<TestType> gpu1(cpu1); // copy construct (std::vector)
    tassert(data_match(cpu1, gpu1));

    gpu::vector<TestType> gpu2(std::vector<TestType>{1, 2}); // move construct
    tassert(data_match(std::vector<TestType>{1, 2}, gpu2));

    std::vector<TestType> cpu2;
    cpu2 = gpu0 = gpu2 = cpu1; // copy assignments
    tassert(data_match(cpu1, gpu2));
    tassert(data_match(gpu2, gpu0));
    tassert(data_match(gpu0, cpu2));

    gpu0 = std::vector<TestType>{1, 2, 3, 4, 5, 6}; // move assign (std::vector)
    tassert(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu0));

    gpu1 = std::move(gpu0); // move assign
    tassert(gpu0.size() == 0);
    tassert(gpu0.data() == nullptr);
    tassert(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu1));

    gpu1.clear();
    tassert(gpu1.size() == 0);
    tassert(gpu1.data() == nullptr);
    tassert(gpu1.empty());

    cpu1 = {1, 2, 3, 4, 5, 6, 7, 8};
    gpu0 = cpu1;
    gpu::vector<TestType> gpu3(std::move(gpu0)); // move construct
    tassert(gpu0.empty());
    tassert(data_match(cpu1, gpu3));
  }
}
#endif

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("accelerated compute", " CPU and GPU capabilities");

  {
    current_test name_("initialize compute");
    init_compute();
  }

#ifdef ASGARD_USE_GPU
  gpu_vector_tests<int>();
  gpu_vector_tests<double>();
  gpu_vector_tests<float>();
#endif

  return 0;
}
