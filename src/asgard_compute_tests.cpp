#include "asgard.hpp"

#include "asgard_test_macros.hpp"

using namespace asgard;

#ifdef ASGARD_USE_GPU
template<typename P>
struct overloads {
  P test1(gpu::vector<P>) { return P{0}; }
  int test1(gpu::wrap_array<P>) { return 1; }

  P test2(gpu::wrap_array<P>) { return P{0}; }
  int test2(std::vector<P>) { return 1; }

  P test3(gpu::wrap_array<const P>) { return P{0}; }
  int test3(std::vector<P>) { return 1; }

  // technically should never have both const and non-const variants
  P test4(gpu::wrap_array<const P>) { return P{0}; }
  int test4(gpu::wrap_array<P>) { return 1; }
};

void wrap_array_compile_test() {
  gpu::vector<double> dx;
  gpu::vector<float> fx;
  {
    gpu::wrap_array<double> wx = dx;
    gpu::wrap_array<const double> cwx = dx;
    cwx = wx;
  }{
    gpu::wrap_array<float> wx = fx;
    gpu::wrap_array<const float> cwx = fx;
    cwx = wx;
  }{
    current_test<double> name_("wrap-array");
    overloads<double> funcs;
    auto t1 = funcs.test1(dx);
    static_assert(std::is_same_v<std::decay_t<decltype(t1)>, double>, "std vs gpu vector");
    auto t2 = funcs.test2(dx);
    static_assert(std::is_same_v<std::decay_t<decltype(t2)>, double>, "auto convert to wrap-array");
    auto t3 = funcs.test3(dx);
    static_assert(std::is_same_v<std::decay_t<decltype(t3)>, double>, "auto convert to const-wrap-array");

    std::ignore = funcs.test3(gpu::wrap_array<double>{dx}); // test converting non-const to const

    gpu::vector<double> const &ref_dx = dx;
    auto t4 = funcs.test4(ref_dx);
    static_assert(std::is_same_v<std::decay_t<decltype(t4)>, double>, "respect const");
  }{
    current_test<float> name_("wrap-array");
    overloads<float> funcs;
    auto t1 = funcs.test1(fx);
    static_assert(std::is_same_v<std::decay_t<decltype(t1)>, float>, "std vs gpu vector");
    auto t2 = funcs.test2(fx);
    static_assert(std::is_same_v<std::decay_t<decltype(t2)>, float>, "auto convert to wrap-array");
    auto t3 = funcs.test3(fx);
    static_assert(std::is_same_v<std::decay_t<decltype(t3)>, float>, "auto convert to const-wrap-array");

    gpu::vector<float> const &ref_fx = fx;
    auto t4 = funcs.test4(ref_fx);
    static_assert(std::is_same_v<std::decay_t<decltype(t4)>, float>, "respect const");
  }
}

template<typename T, typename vec_type_2>
bool data_match(std::vector<T> const &v1, vec_type_2 const &v2)
{
  if (static_cast<int64_t>(v1.size()) != static_cast<int64_t>(v2.size()))
    return false;
  static_assert(std::is_same_v<T, typename vec_type_2::value_type>);
  auto y = v2.copy_to_host();
  static_assert(std::is_same_v<decltype(y), std::vector<typename vec_type_2::value_type>>);
  if (v1.size() != y.size()) // something happened during copy
    return false;
  for (size_t i = 0; i < v1.size(); i++)
    if (v1[i] != y[i]) // this checks data copies, so it's OK for floating point numbers
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
    (gpu0 = gpu2 = cpu1).copy_to_host(cpu2); // copy assignments
    tassert(data_match(cpu1, gpu2));
    tassert(data_match(gpu2.copy_to_host(), gpu0));
    tassert(data_match(cpu2, gpu0));

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
  wrap_array_compile_test();
  gpu_vector_tests<int>();
  gpu_vector_tests<double>();
  gpu_vector_tests<float>();
#endif

  return 0;
}
