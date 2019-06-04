#include "apply_fmwt.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>

TEMPLATE_TEST_CASE("apply_fmwt", "[apply_fmwt]", double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      auto const f1       = *first_it++;
      auto const diff     = std::abs(f1 - second_elem);
      auto const abs_diff = std::abs(f1 - second_elem);
      REQUIRE(diff < std::numeric_limits<TestType>::epsilon() * 1.0e5 *
                         std::max(std::abs(f1), std::abs(second_elem)));
    });
  };

  // Testing of various apply fmwt methods
  // for two size arrays generated (random [0,1]) matrix
  // is read in and used as the coefficient matrix
  // the optimized methods (method 2 and 3) are then compared
  // to the full matrix multiplication (method 1)
  SECTION("Apply fmwt test set 1 - kdeg=2 lev=2")
  {
    int kdeg             = 2;
    int lev              = 2;
    std::string out_base = "../testing/generated-inputs/apply_fmwt/";

    std::string mat1_string = out_base + "mat1_k1_lev2.dat";

    dimension const dim = make_PDE<TestType>(PDE_opts::continuity_1, lev, kdeg)
                              ->get_dimensions()[0];

    fk::matrix<TestType> const fmwt = operator_two_scale<TestType>(kdeg, lev);

    fk::matrix<TestType> mat1 =
        fk::matrix<TestType>(read_matrix_from_txt_file(mat1_string));

    int isLeft  = 1;
    int isTrans = 0;
    int method  = 2;
    auto const productLeft1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productLeft2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply left - method 2")
    {
      relaxed_comparison(productLeft1, productLeft2);
    }

    isLeft  = 1;
    isTrans = 1;
    method  = 2;
    auto const productLeftTrans1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productLeftTrans2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply left transpose - method 2")
    {
      relaxed_comparison(productLeftTrans1, productLeftTrans2);
    }

    isLeft  = 0;
    isTrans = 0;
    auto const productRight1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productRight2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);

    SECTION("degree = 2, lev 2 fmwt apply right - method 2")
    {
      relaxed_comparison(productRight1, productRight2);
    }

    isLeft  = 0;
    isTrans = 1;
    auto const productRightTrans1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productRightTrans2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply right transpose - method 2")
    {
      relaxed_comparison(productRightTrans1, productRightTrans2);
    }

    method  = 3;
    isLeft  = 1;
    isTrans = 0;
    auto const productLeft3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply left method 3")
    {
      relaxed_comparison(productLeft1, productLeft3);
    }

    isLeft  = 1;
    isTrans = 1;
    auto const productLeftTrans3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply left method 3")
    {
      relaxed_comparison(productLeftTrans1, productLeftTrans3);
    }

    isLeft  = 0;
    isTrans = 0;
    auto const productRight3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply right method 3")
    {
      relaxed_comparison(productRight1, productRight3);
    }

    isLeft  = 0;
    isTrans = 1;
    auto const productRightTrans3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 2, lev 2 fmwt apply right trans method 3")
    {
      relaxed_comparison(productRightTrans1, productRightTrans3);
    }
  }

  SECTION("Apply fmwt test set 2 - kdeg=4 lev=5")
  {
    int kdeg             = 4;
    int lev              = 5;
    std::string out_base = "../testing/generated-inputs/apply_fmwt/";

    std::string mat1_string = out_base + "mat1.dat";

    dimension const dim = make_PDE<TestType>(PDE_opts::continuity_1, lev, kdeg)
                              ->get_dimensions()[0];

    fk::matrix<TestType> const fmwt = operator_two_scale<TestType>(kdeg, lev);

    fk::matrix<TestType> mat1 =
        fk::matrix<TestType>(read_matrix_from_txt_file(mat1_string));

    int isLeft  = 1;
    int isTrans = 0;
    int method  = 2;
    auto const productLeft1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productLeft2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply left - method 2")
    {
      relaxed_comparison(productLeft1, productLeft2);
    }

    isLeft  = 1;
    isTrans = 1;
    method  = 2;
    auto const productLeftTrans1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productLeftTrans2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply left transpose - method 2")
    {
      relaxed_comparison(productLeftTrans1, productLeftTrans2);
    }

    isLeft  = 0;
    isTrans = 0;
    auto const productRight1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productRight2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply right - method 2")
    {
      relaxed_comparison(productRight1, productRight2);
    }

    isLeft  = 0;
    isTrans = 1;
    auto const productRightTrans1 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, 1);
    auto const productRightTrans2 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply right transpose - method 2")
    {
      relaxed_comparison(productRightTrans1, productRightTrans2);
    }

    method  = 3;
    isLeft  = 1;
    isTrans = 0;
    auto const productLeft3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply left - method 3")
    {
      relaxed_comparison(productLeft1, productLeft3);
    }

    isLeft  = 1;
    isTrans = 1;
    auto const productLeftTrans3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply left - method 3")
    {
      relaxed_comparison(productLeftTrans1, productLeftTrans3);
    }

    isLeft  = 0;
    isTrans = 0;
    auto const productRight3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply right - method 3")
    {
      relaxed_comparison(productRight1, productRight3);
    }

    isLeft  = 0;
    isTrans = 1;
    auto const productRightTrans3 =
        apply_fmwt<TestType>(fmwt, mat1, kdeg, lev, isLeft, isTrans, method);
    SECTION("degree = 4, lev 5 fmwt apply right trans - method 3")
    {
      relaxed_comparison(productRightTrans1, productRightTrans3);
    }
  }
}
