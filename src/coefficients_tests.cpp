#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

template<typename P>
static inline void relaxed_comparison(fk::matrix<double> const first,
                                      fk::matrix<double> const second)
{
  auto first_it = first.begin();

  std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
    REQUIRE(
        Approx(*first_it++).epsilon(std::numeric_limits<P>::epsilon() * 1e3) ==
        second_elem);
  });
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto const continuity1 = make_PDE<TestType>(PDE_opts::continuity_1);
  std::string const filename =
      "../testing/generated-inputs/coefficients/continuity1_coefficients.dat";
  fk::matrix<double> const gold = read_matrix_from_txt_file(filename);
  fk::matrix<double> const test = generate_coefficients<TestType>(
      continuity1->get_dimensions()[0], continuity1->get_terms()[0][0], 0.0);
  relaxed_comparison<TestType>(gold, test);
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", double, float)
{
  int const level     = 4;
  int const degree    = 3;
  TestType const time = 1.0;
  auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
  std::string const filename_base =
      "../testing/generated-inputs/coefficients/continuity2_coefficients_l" +
      std::to_string(level) + "_d" + std::to_string(degree) + "_";
  for (int t = 0; t < pde->num_terms; ++t)
  {
    for (int d = 0; d < pde->num_dims; ++d)
    {
      std::string const filename = filename_base + std::to_string(t + 1) + "_" +
                                   std::to_string(d + 1) + ".dat";
      fk::matrix<double> const gold = read_matrix_from_txt_file(filename);
      fk::matrix<double> const test = generate_coefficients<TestType>(
          pde->get_dimensions()[d], pde->get_terms()[t][d], time);
      relaxed_comparison<TestType>(gold, test);
    }
  }
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", double, float)
{
  int const level     = 3;
  int const degree    = 5;
  TestType const time = 1.0;
  auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
  std::string const filename_base =
      "../testing/generated-inputs/coefficients/continuity3_coefficients_l" +
      std::to_string(level) + "_d" + std::to_string(degree) + "_";
  for (int t = 0; t < pde->num_terms; ++t)
  {
    for (int d = 0; d < pde->num_dims; ++d)
    {
      std::string const filename = filename_base + std::to_string(t + 1) + "_" +
                                   std::to_string(d + 1) + ".dat";
      fk::matrix<double> const gold = read_matrix_from_txt_file(filename);
      fk::matrix<double> const test = generate_coefficients<TestType>(
          pde->get_dimensions()[d], pde->get_terms()[t][d], time);
      relaxed_comparison<TestType>(gold, test);
    }
  }
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", double, float)
{
  int const level     = 2;
  int const degree    = 4;
  TestType const time = 1.0;
  auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
  std::string const filename_base =
      "../testing/generated-inputs/coefficients/continuity6_coefficients_l" +
      std::to_string(level) + "_d" + std::to_string(degree) + "_";
  for (int t = 0; t < pde->num_terms; ++t)
  {
    for (int d = 0; d < pde->num_dims; ++d)
    {
      std::string const filename = filename_base + std::to_string(t + 1) + "_" +
                                   std::to_string(d + 1) + ".dat";
      fk::matrix<double> const gold = read_matrix_from_txt_file(filename);
      fk::matrix<double> const test = generate_coefficients<TestType>(
          pde->get_dimensions()[d], pde->get_terms()[t][d], time);
      relaxed_comparison<TestType>(gold, test);
    }
  }
}
