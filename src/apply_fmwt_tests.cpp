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

    std::string mat1_string    = out_base + "mat1.dat";
    std::string fmwt_string    = out_base + "fmwt_k4_lev5.dat";

    fk::matrix<TestType> mat1 =
        fk::matrix<TestType>(read_matrix_from_txt_file(mat1_string));
    fk::matrix<TestType> fmwt =
        fk::matrix<TestType>(read_matrix_from_txt_file(fmwt_string));
    
    mat1.print("mat1");	
    fmwt.print("fmwt");	
  
    fk::matrix<TestType> product = fmwt*mat1;
    auto const product2 = apply_fmwt<TestType>(fmwt,mat1);
    //fk::matrix<TestType> product2 = apply_fmwt(fmwt,mat1);
    product.print("product");	
    
    SECTION("degree = 4, lev 5 fmwt apply") { relaxed_comparison(product, product2); }
  }

}
