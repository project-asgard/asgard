#include "tests_general.hpp"

#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT

#ifdef ASGARD_USE_CUDA
#define mtest_precs                                     \
  (double, resource::host), (double, resource::device), \
      (float, resource::host), (float, resource::device)
#else
#define mtest_precs (double, resource::host), (float, resource::host)
#endif

#else

#ifdef ASGARD_USE_CUDA
#define mtest_precs (double, resource::host), (double, resource::device)
#else
#define mtest_precs (double, resource::host)
#endif

#endif
#else

#ifdef ASGARD_USE_CUDA
#define mtest_precs (float, resource::host), (float, resource::device)
#else
#define mtest_precs (float, resource::host)
#endif

#endif

static auto const transformations_base_dir = gold_base_dir / "transformations";

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

template<typename P>
void test_multiwavelet_gen(int const degree, P const tol_factor)
{
  std::string const out_base = "multiwavelet_" + std::to_string(degree + 1) + "_";

  auto const [m_h0, m_h1, m_g0, m_g1] = generate_multi_wavelets<P>(degree);

  auto const [h0, h1, g0, g1, scale_co] = [&out_base, degree]() {
    auto const h0_string = transformations_base_dir / (out_base + "h0.dat");
    auto const h1_string = transformations_base_dir / (out_base + "h1.dat");
    auto const g0_string = transformations_base_dir / (out_base + "g0.dat");
    auto const g1_string = transformations_base_dir / (out_base + "g1.dat");

    if (degree < 1)
    {
      auto const h0_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h0_string))}};
      auto const h1_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h1_string))}};
      auto const g0_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g0_string))}};
      auto const g1_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g1_string))}};
      return std::array<fk::matrix<P>, 5>{h0_out, h1_out, g0_out, g1_out};
    }
    else
    {
      fk::matrix<P> const h0_out = read_matrix_from_txt_file<P>(h0_string);
      fk::matrix<P> const h1_out = read_matrix_from_txt_file<P>(h1_string);
      fk::matrix<P> const g0_out = read_matrix_from_txt_file<P>(g0_string);
      fk::matrix<P> const g1_out = read_matrix_from_txt_file<P>(g1_string);
      return std::array<fk::matrix<P>, 5>{h0_out, h1_out, g0_out, g1_out};
    }
  }();

  rmse_comparison(h0, m_h0, tol_factor);
  rmse_comparison(h1, m_h1, tol_factor);
  rmse_comparison(g0, m_g0, tol_factor);
  rmse_comparison(g1, m_g1, tol_factor);
}

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", test_precs)
{
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("Multiwavelet generation, degree = 0")
  {
    int const degree = 0;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 1")
  {
    int const degree = 1;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 2")
  {
    int const degree = 2;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 3")
  {
    int const degree = 3;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }
}

template<typename P>
void test_operator_two_scale(int const levels, int const degree)
{
  auto filename = transformations_base_dir /
                  ("operator_two_scale_" + std::to_string(degree + 1) + "_" +
                   std::to_string(levels) + ".dat");
  fk::matrix<P> const gold = read_matrix_from_txt_file<P>(filename);
  fk::matrix<P> const test = operator_two_scale<P>(degree, levels);

  auto constexpr tol_factor = get_tolerance<P>(100);

  rmse_comparison(gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("operator_two_scale function working appropriately",
                   "[transformations]", test_precs)
{
  SECTION("operator_two_scale(2, 1)")
  {
    int const degree = 1;
    int const levels = 2;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(3, 1)")
  {
    int const degree = 1;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(3, 3)")
  {
    int const degree = 3;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(5, 4)")
  {
    int const degree = 4;
    int const levels = 5;
    test_operator_two_scale<TestType>(levels, degree);
  }

  SECTION("operator_two_scale(6, 2)")
  {
    int const degree = 1;
    int const levels = 6;
    test_operator_two_scale<TestType>(levels, degree);
  }
}
