
#include "asgard_resources.hpp"
#include "asgard_vector.hpp"
#include "tests_general.hpp"

using namespace asgard;

TEMPLATE_TEST_CASE("device functions", "[resources]", test_precs, int)
{
#ifdef ASGARD_USE_CUDA
  SECTION("allocate_device, delete_device")
  {
    TestType *ptr = nullptr;
    int64_t num_elems{2};
    fk::allocate_device(ptr, num_elems, true);
    REQUIRE(ptr != nullptr);
    fk::delete_device(ptr);
    REQUIRE(ptr == nullptr);
  }

  SECTION("copy_to_device, copy_to_host")
  {
    TestType *b = nullptr;
    int64_t num_elems{2};
    fk::allocate_device(b, num_elems, false);
    fk::vector<TestType> const a{1, 2};
    REQUIRE(a.size() == num_elems);
    fk::vector<TestType> c(num_elems);
    fk::copy_to_device(b, a.data(), num_elems);
    fk::copy_to_host(c.data(), b, num_elems);
    fk::delete_device(b);
    REQUIRE(c == a);
  }

  SECTION("copy_to_device, copy_on_device, copy_to_host")
  {
    TestType *b = nullptr;
    TestType *c = nullptr;
    int64_t num_elems{2};
    fk::allocate_device(b, num_elems, false);
    REQUIRE(b != nullptr);
    fk::allocate_device(c, num_elems, false);
    REQUIRE(c != nullptr);
    fk::vector<TestType> const a{1, 2};
    REQUIRE(a.size() == num_elems);
    fk::vector<TestType> d(num_elems);
    fk::copy_to_device(b, a.data(), num_elems);
    fk::copy_on_device(c, b, num_elems);
    fk::copy_to_host(d.data(), c, num_elems);
    fk::delete_device(b);
    REQUIRE(b == nullptr);
    fk::delete_device(c);
    REQUIRE(c == nullptr);
    REQUIRE(d == a);
  }
#endif
}

#ifdef ASGARD_USE_CUDA
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
  for(size_t i = 0; i < x.size(); i++)
    if (x[i] != y[i]) // this checks data copies, so it's OK for floating point numbers
      return false;
  return true;
}

TEMPLATE_TEST_CASE("gpu::vector", "[gpu::vector]", test_precs, int)
{
  SECTION("allocate_device, delete_device")
  {
    gpu::vector<TestType> gpu0; // make empty
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);
    REQUIRE(gpu0.empty());

    gpu0.resize(10); // resize
    REQUIRE(gpu0.size() == 10);
    REQUIRE(gpu0.data() != nullptr);
    REQUIRE(not gpu0.empty());

    gpu0 = gpu::vector<TestType>(); // move-assign
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);

    std::vector<TestType> cpu1 = {1, 2, 3, 4};
    gpu::vector<TestType> gpu1(cpu1); // copy construct (std::vector)
    REQUIRE(data_match(cpu1, gpu1));

    gpu::vector<TestType> gpu2(std::vector<TestType>{1, 2}); // move construct
    REQUIRE(data_match(std::vector<TestType>{1, 2}, gpu2));

    std::vector<TestType> cpu2;
    cpu2 = gpu0 = gpu2 = cpu1; // copy assignments
    REQUIRE(data_match(cpu1, gpu2));
    REQUIRE(data_match(gpu2, gpu0));
    REQUIRE(data_match(gpu0, cpu2));

    gpu0 = std::vector<TestType>{1, 2, 3, 4, 5, 6}; // move assign (std::vector)
    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu0));

    gpu1 = std::move(gpu0); // move assign
    REQUIRE(gpu0.size() == 0);
    REQUIRE(gpu0.data() == nullptr);
    REQUIRE(data_match(std::vector<TestType>{1, 2, 3, 4, 5, 6}, gpu1));

    gpu1.clear();
    REQUIRE(gpu1.size() == 0);
    REQUIRE(gpu1.data() == nullptr);
    REQUIRE(gpu1.empty());

    cpu1 = {1, 2, 3, 4, 5, 6, 7, 8};
    gpu0 = cpu1;
    gpu::vector<TestType> gpu3(std::move(gpu0)); // move construct
    REQUIRE(gpu0.empty());
    REQUIRE(data_match(cpu1, gpu3));
  }
}
#endif
