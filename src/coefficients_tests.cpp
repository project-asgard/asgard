#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e2) ==
              second_elem);
    });
  };
  auto continuity1 = make_PDE<TestType>(PDE_opts::continuity_1);
  std::string filename =
      "../testing/generated-inputs/coefficients/continuity1_coefficients.dat";
  fk::matrix<TestType> const gold = read_matrix_from_txt_file(filename);
  fk::matrix<TestType> const test = generate_coefficients<TestType>(
      continuity1->dimensions[0], continuity1->terms[0][0], 0.0);
  relaxed_comparison(gold, test);
}
