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
