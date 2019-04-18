#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "apply_fmwt.hpp"
#include <numeric>

TEMPLATE_TEST_CASE("apply_fmwt", "[apply_fmwt]", double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e3) ==
              second_elem);
    });
  };

  SECTION("Apply fmwt test 1")
  {
    int const degree = 3;
    std::string out_base =
        "../testing/generated-inputs/apply_fmwt/";

    std::string h0_string    = out_base + "matrix.dat";

    fk::matrix<TestType> h0 =
        fk::matrix<TestType>(read_matrix_from_txt_file(h0_string));
    
    h0.print("h0");	
    //std::string out_base =
    //    "../testing/generated-inputs/apply_fmwt";
    //std::string matrix_string    = out_base + "/matrix.dat";
    //
    //TestType mat = static_cast<TestType>(read_scalar_from_txt_file(matrix_string));
    //
    //fk::matrix<TestType> mat1 =
    //    fk::matrix<TestType>(read_matrix_from_txt_file(matrix_string));
    //mat1.print("mat1");	
  }

}
