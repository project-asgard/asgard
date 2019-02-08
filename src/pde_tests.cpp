#include "pde.hpp"

#include "tests_general.hpp"
#include <vector>

// FIXME need to update these
TEST_CASE("testing pde implementations", "[pde]")
{
  SECTION("vlasov7 functions against golden values")
  {
    auto pde = make_PDE<float>(PDE_opts::continuity_1);
    /*auto pde = make_PDE<float>(PDE_opts::vlasov7);

    std::vector<float> x = {1.0, 2.0, 3.0};
    float t              = 3.0;

    std::vector<float> fx = {0.0, -2.0, -6.0};
    std::vector<float> fv = {0.0, 0.0, 0.0};

    std::vector<float> ex = fx;
    std::vector<float> ev = {-24.0, -21.0, -16.0};
    float et              = t;

    std::vector<float> s0_x = fx;
    std::vector<float> s0_v = ev;
    float s0_t              = 1.0;

    std::vector<float> s1_x = {-1.0, -3.0, -5.0};
    std::vector<float> s1_v = {-24.0, -42.0, -48.0};
    float s1_t              = t;

    std::vector<float> s2_x = fx;
    std::vector<float> s2_v = {2.0, 4.0, 6.0};
    float s2_t              = t;

    REQUIRE(fx == pde->initial_condition_funcs()[0](x));
    REQUIRE(fv == pde->initial_condition_funcs()[1](x));

    REQUIRE(ex == pde->exact_vector_funcs()[0](x));
    REQUIRE(ev == pde->exact_vector_funcs()[1](x));
    REQUIRE(et == pde->exact_scalar_func()(t));

    REQUIRE(s0_x == pde->source_vector_funcs()[0][0](x));
    REQUIRE(s0_v == pde->source_vector_funcs()[0][1](x));
    REQUIRE(s0_t == pde->source_scalar_funcs()[0](t));

    REQUIRE(s1_x == pde->source_vector_funcs()[1][0](x));
    REQUIRE(s1_v == pde->source_vector_funcs()[1][1](x));
    REQUIRE(s1_t == pde->source_scalar_funcs()[1](t));

    REQUIRE(s2_x == pde->source_vector_funcs()[2][0](x));
    REQUIRE(s2_v == pde->source_vector_funcs()[2][1](x));
    REQUIRE(s2_t == pde->source_scalar_funcs()[2](t));
    */
  }
}
