#include "asgard.hpp"

#include <catch2/catch_all.hpp>

template<typename P>
void test_compile()
{
  asgard::prog_opts opts;
  opts.pde_choice   = asgard::PDE_opts::custom;
  opts.start_levels = {2, 2};

  asgard::PDE<P> empty_pde;
  asgard::ignore(empty_pde);

  auto diff_pde =
        asgard::make_custom_pde<asgard::PDE_diffusion_2d<P>>(opts);

  static_assert(std::is_same_v<decltype(diff_pde),
                               std::unique_ptr<asgard::PDE<P>>>);
}

TEST_CASE("compile time testing", "[main]")
{
  // we could put some tests here
  // right now we just look at compile tests
  // the following declarations should compile

#ifdef ASGARD_ENABLE_DOUBLE
  test_compile<double>();
#endif

#ifdef ASGARD_ENABLE_FLOAT
  test_compile<float>();
#endif

  REQUIRE(true);
}
