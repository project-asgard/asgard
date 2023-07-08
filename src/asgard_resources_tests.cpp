
#include "tensors.hpp"
#include "tests_general.hpp"

using namespace asgard;

#ifdef ASGARD_USE_CUDA
TEMPLATE_TEST_CASE("device functions", "[resources]", test_precs, int)
{
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
}
#endif

TEMPLATE_TEST_CASE("Copy matrix functions for various resources", "[resources]",
                   test_precs, int)
{
  fk::matrix<TestType> const a{{1, 2}, {3, 4}};

  SECTION("check host -> host")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    copy_matrix(b, a);
    REQUIRE(b == a);
  }

  SECTION("check host -> host view")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    fk::matrix<TestType, mem_type::view> b_v(b);
    copy_matrix(b_v, a);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check host view -> host")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType> b(a.nrows(), a.ncols());

    copy_matrix(b, a_v);
    REQUIRE(b == a_v);
    REQUIRE(b == a);
  }

#ifdef ASGARD_USE_CUDA
  fk::matrix<TestType, mem_type::owner, resource::device> a_d(
      a.clone_onto_device());
  SECTION("check host -> device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host -> device view")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    fk::matrix<TestType, mem_type::view, resource::device> b_v(b);
    copy_matrix(b_v, a);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host view -> device")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType, mem_type::owner, resource::device> b_d(a.nrows(),
                                                                a.ncols());

    copy_matrix(b_d, a_v);
    REQUIRE(b_d.clone_onto_host() == a_v);
    REQUIRE(b_d.clone_onto_host() == a);
  }

  SECTION("check device -> host")
  {
    fk::matrix<TestType> b(a.nrows(), a.ncols());
    copy_matrix(b, a_d);
    REQUIRE(b == a);
  }

  SECTION("check device -> host view")
  {
    fk::matrix<TestType, mem_type::owner> b(a.nrows(), a.ncols());
    fk::matrix<TestType, mem_type::view> b_v(b);
    copy_matrix(b_v, a_d);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check device view -> host")
  {
    fk::matrix<TestType, mem_type::const_view> a_v(a);
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());

    copy_matrix(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a_d);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device view")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    fk::matrix<TestType, mem_type::view, resource::device> b_v(b);
    copy_matrix(b_v, a_d);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device view -> device")
  {
    fk::matrix<TestType, mem_type::const_view, resource::device> a_v(a_d);
    fk::matrix<TestType, mem_type::owner, resource::device> b(a.nrows(),
                                                              a.ncols());
    copy_matrix(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v.clone_onto_host());
    REQUIRE(b.clone_onto_host() == a);
  }
#endif
}

TEMPLATE_TEST_CASE("Copy vector functions for various resources", "[resources]",
                   test_precs, int)
{
  fk::vector<TestType> const a{1, 2, 3, 4};

  SECTION("check host -> host")
  {
    fk::vector<TestType> b(a.size());
    copy_vector(b, a);
    REQUIRE(b == a);
  }

  SECTION("check host -> host view")
  {
    fk::vector<TestType> b(a.size());
    fk::vector<TestType, mem_type::view> b_v(b);
    copy_vector(b_v, a);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check host view -> host")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType> b(a.size());
    copy_vector(b, a_v);
    REQUIRE(b == a_v);
    REQUIRE(b == a);
  }

#ifdef ASGARD_USE_CUDA
  fk::vector<TestType, mem_type::owner, resource::device> a_d(
      a.clone_onto_device());
  SECTION("check host -> device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host -> device view")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    fk::vector<TestType, mem_type::view, resource::device> b_v(b);
    copy_vector(b_v, a);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check host view -> device")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType, mem_type::owner, resource::device> b_d(a.size());
    copy_vector(b_d, a_v);
    REQUIRE(b_d.clone_onto_host() == a_v);
    REQUIRE(b_d.clone_onto_host() == a);
  }

  SECTION("check device -> host")
  {
    fk::vector<TestType> b(a.size());
    copy_vector(b, a_d);
    REQUIRE(b == a);
  }

  SECTION("check device -> host view")
  {
    fk::vector<TestType, mem_type::owner> b(a.size());
    fk::vector<TestType, mem_type::view> b_v(b);
    copy_vector(b_v, a_d);
    REQUIRE(b_v == a);
    REQUIRE(b == a);
  }

  SECTION("check device view -> host")
  {
    fk::vector<TestType, mem_type::const_view> a_v(a);
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());

    copy_vector(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a_d);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device -> device view")
  {
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    fk::vector<TestType, mem_type::view, resource::device> b_v(b);
    copy_vector(b_v, a_d);
    REQUIRE(b_v.clone_onto_host() == a);
    REQUIRE(b.clone_onto_host() == a);
  }

  SECTION("check device view -> device")
  {
    fk::vector<TestType, mem_type::const_view, resource::device> a_v(a_d);
    fk::vector<TestType, mem_type::owner, resource::device> b(a.size());
    copy_vector(b, a_v);
    REQUIRE(b.clone_onto_host() == a_v.clone_onto_host());
    REQUIRE(b.clone_onto_host() == a);
  }
#endif
}
