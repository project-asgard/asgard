#include "kronmult_cuda.hpp"
#include "tests_general.hpp"

template<typename P>
void test_kronmult_staging(int const num_elems, int const num_copies)
{
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-4.0, 4.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  fk::vector<P, mem_type::owner, resource::device> const x = [gen,
                                                              num_elems]() {
    fk::vector<P> x(num_elems);
    std::generate(x.begin(), x.end(), gen);
    return x.clone_onto_device();
  }();
  fk::vector<P> const gold(x.clone_onto_host());

  fk::vector<P, mem_type::owner, resource::device> dest(x.size() * num_copies);

  stage_inputs_kronmult(x.data(), dest.data(), x.size(), num_copies);

  for (auto i = 0; i < num_copies; ++i)
  {
    fk::vector<P, mem_type::const_view, resource::device> const dest_view(
        dest, i * x.size(), (i + 1) * x.size() - 1);
    fk::vector<P> const test(dest_view.clone_onto_host());
    REQUIRE(test == gold);
  }
}

TEMPLATE_TEST_CASE("staging kernel", "[kronmult_cuda]", float, double)
{
  SECTION("1 elems, 1 copies")
  {
    auto const num_elems  = 1;
    auto const num_copies = 1;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("1001 elems, 1 copies")
  {
    auto const num_elems  = 1001;
    auto const num_copies = 1;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("1 elems, 500 copies")
  {
    auto const num_elems  = 1;
    auto const num_copies = 500;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }

  SECTION("101010 elems, 555 copies")
  {
    auto const num_elems  = 101010;
    auto const num_copies = 555;
    test_kronmult_staging<TestType>(num_elems, num_copies);
  }
}
