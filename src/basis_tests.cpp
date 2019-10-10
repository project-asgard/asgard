#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>
#include <random>

auto const tol_scale = 1e3;

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{

  SECTION("Multiwavelet generation, degree = 1")
  {
    int const degree = 1;
    std::string out_base =
        "../testing/generated-inputs/transformations/multiwavelet_" +
        std::to_string(degree) + "_";
    std::string h0_string    = out_base + "h0.dat";
    std::string h1_string    = out_base + "h1.dat";
    std::string g0_string    = out_base + "g0.dat";
    std::string g1_string    = out_base + "g1.dat";
    std::string phi_string   = out_base + "phi_co.dat";
    std::string scale_string = out_base + "scale_co.dat";

    TestType h0 = static_cast<TestType>(read_scalar_from_txt_file(h0_string));
    TestType h1 = static_cast<TestType>(read_scalar_from_txt_file(h1_string));
    TestType g0 = static_cast<TestType>(read_scalar_from_txt_file(g0_string));
    TestType g1 = static_cast<TestType>(read_scalar_from_txt_file(g1_string));
    fk::matrix<TestType> phi_co =
        fk::matrix<TestType>(read_matrix_from_txt_file(phi_string));
    TestType scale_co =
        static_cast<TestType>(read_scalar_from_txt_file(scale_string));

    auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
        generate_multi_wavelets<TestType>(degree);

    SECTION("degree = 1, h0") { REQUIRE(Approx(h0) == m_h0(0, 0)); }

    SECTION("degree = 1, h1") { REQUIRE(Approx(h1) == m_h1(0, 0)); }
    SECTION("degree = 1, g0") { REQUIRE(Approx(g0) == m_g0(0, 0)); }
    SECTION("degree = 1, g1") { REQUIRE(Approx(g1) == m_g1(0, 0)); }
    SECTION("degree = 1, phi_co")
    {
      relaxed_comparison(phi_co, m_phi_co, tol_scale);
    }
    SECTION("degree = 1, scale_co")
    {
      REQUIRE(Approx(scale_co) == m_scale_co(0, 0));
    }
  }

  SECTION("Multiwavelet generation, degree = 3")
  {
    int const degree = 3;
    std::string out_base =
        "../testing/generated-inputs/transformations/multiwavelet_" +
        std::to_string(degree) + "_";

    std::string h0_string    = out_base + "h0.dat";
    std::string h1_string    = out_base + "h1.dat";
    std::string g0_string    = out_base + "g0.dat";
    std::string g1_string    = out_base + "g1.dat";
    std::string phi_string   = out_base + "phi_co.dat";
    std::string scale_string = out_base + "scale_co.dat";

    fk::matrix<TestType> h0 =
        fk::matrix<TestType>(read_matrix_from_txt_file(h0_string));
    fk::matrix<TestType> h1 =
        fk::matrix<TestType>(read_matrix_from_txt_file(h1_string));
    fk::matrix<TestType> g0 =
        fk::matrix<TestType>(read_matrix_from_txt_file(g0_string));
    fk::matrix<TestType> g1 =
        fk::matrix<TestType>(read_matrix_from_txt_file(g1_string));
    fk::matrix<TestType> phi_co =
        fk::matrix<TestType>(read_matrix_from_txt_file(phi_string));
    fk::matrix<TestType> scale_co =
        fk::matrix<TestType>(read_matrix_from_txt_file(scale_string));

    auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
        generate_multi_wavelets<TestType>(degree);

    SECTION("degree = 3, h0") { relaxed_comparison(h0, m_h0, tol_scale); }
    SECTION("degree = 3, h1") { relaxed_comparison(h1, m_h1, tol_scale); }
    SECTION("degree = 3, g0") { relaxed_comparison(g0, m_g0, tol_scale); }
    SECTION("degree = 3, g1") { relaxed_comparison(g1, m_g1, tol_scale); }
    SECTION("degree = 3, phi_co")
    {
      relaxed_comparison(phi_co, m_phi_co, tol_scale);
    }
    SECTION("degree = 3, scale_co")
    {
      relaxed_comparison(scale_co, m_scale_co);
    }
  }
}

// FIXME we are still off after 12 dec places or so in double prec.
// need to think about these precision issues some more
TEMPLATE_TEST_CASE("operator_two_scale function working appropriately",
                   "[transformations]", double)
{

  SECTION("operator_two_scale(2, 2)")
  {
    int const degree = 2;
    int const levels = 2;

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::matrix<TestType> const gold =
        fk::matrix<TestType>(read_matrix_from_txt_file(
            "../testing/generated-inputs/transformations/operator_two_scale_" +
            std::to_string(degree) + "_" + std::to_string(levels) + ".dat"));
    fk::matrix<TestType> const test =
        operator_two_scale<TestType>(dim.get_degree(), dim.get_level());
    relaxed_comparison(gold, test);
  }

  SECTION("operator_two_scale(2, 3)")
  {
    int const degree = 2;
    int const levels = 3;

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::matrix<TestType> const gold =
        fk::matrix<TestType>(read_matrix_from_txt_file(
            "../testing/generated-inputs/transformations/operator_two_scale_" +
            std::to_string(degree) + "_" + std::to_string(levels) + ".dat"));
    fk::matrix<TestType> const test =
        operator_two_scale<TestType>(dim.get_degree(), dim.get_level());
    relaxed_comparison(gold, test);
  }
  SECTION("operator_two_scale(4, 3)")
  {
    int const degree = 4;
    int const levels = 3;

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test =
        operator_two_scale<TestType>(dim.get_degree(), dim.get_level());
    relaxed_comparison(gold, test);
  }
  SECTION("operator_two_scale(5, 5)")
  {
    int const degree = 5;
    int const levels = 5;
    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test =
        operator_two_scale<TestType>(dim.get_degree(), dim.get_level());

    relaxed_comparison(gold, test);
  }

  SECTION("operator_two_scale(2, 6)")
  {
    int const degree = 2;
    int const levels = 6;

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test =
        operator_two_scale<TestType>(dim.get_degree(), dim.get_level());
    relaxed_comparison(gold, test);
  }
}

TEMPLATE_TEST_CASE("apply_fmwt", "[apply_fmwt]", double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      auto const f1 = *first_it++;
      auto tol      = std::numeric_limits<TestType>::epsilon() * 1e3;
      REQUIRE_THAT(f1, Catch::Matchers::WithinAbs(second_elem, tol));
    });
  };

  // Testing of various apply fmwt methods
  // for two size arrays generated (random [0,1]) matrix
  // is read in and used as the coefficient matrix
  // the optimized methods (method 2 and 3) are then compared
  // to the full matrix multiplication (method 1)
  SECTION("Apply fmwt test set 1 - kdeg=2 lev=2")
  {
    int kdeg = 2;
    int lev  = 2;
    int n    = kdeg * pow(2, lev);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    // here is the distro you want to generate random values within
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    fk::matrix<TestType> mat1(n, n);
    std::generate(mat1.begin(), mat1.end(), gen);
    fk::matrix<TestType> const fmwt = operator_two_scale<TestType>(kdeg, lev);

    auto const product_left1 = fmwt * mat1;
    auto const product_left2 = apply_left_fmwt<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 2, lev 2 fmwt apply left - method 2")
    {
      relaxed_comparison(product_left1, product_left2);
    }

    fk::matrix<TestType> fmwt_transpose =
        fk::matrix<TestType>(fmwt).transpose();
    fk::matrix<TestType> product_left_trans1 = fmwt_transpose * mat1;

    auto const product_left_trans2 =
        apply_left_fmwt_transposed<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 2, lev 2 fmwt apply left transpose - method 2")
    {
      relaxed_comparison(product_left_trans1, product_left_trans2);
    }

    auto const product_right1 = mat1 * fmwt;
    auto const product_right2 =
        apply_right_fmwt<TestType>(fmwt, mat1, kdeg, lev);

    SECTION("degree = 2, lev 2 fmwt apply right - method 2")
    {
      relaxed_comparison(product_right1, product_right2);
    }

    auto const product_right_trans1 = mat1 * fmwt_transpose;
    auto const product_right_trans2 =
        apply_right_fmwt_transposed<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 2, lev 2 fmwt apply right transpose - method 2")
    {
      relaxed_comparison(product_right_trans1, product_right_trans2);
    }
  }

  SECTION("Apply fmwt test set 2 - kdeg=4 lev=5")
  {
    int kdeg = 4;
    int lev  = 5;
    int n    = kdeg * pow(2, lev);

    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    // here is the distro you want to generate random values within
    std::uniform_real_distribution<TestType> dist(-2.0, 2.0);
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    fk::matrix<TestType> mat1(n, n);
    std::generate(mat1.begin(), mat1.end(), gen);

    fk::matrix<TestType> const fmwt = operator_two_scale<TestType>(kdeg, lev);

    auto const product_left1 = fmwt * mat1;
    auto const product_left2 = apply_left_fmwt<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 4, lev 5 fmwt apply left - method 2")
    {
      relaxed_comparison(product_left1, product_left2);
    }

    fk::matrix<TestType> fmwt_transpose =
        fk::matrix<TestType>(fmwt).transpose();
    fk::matrix<TestType> product_left_trans1 = fmwt_transpose * mat1;
    auto const product_left_trans2 =
        apply_left_fmwt_transposed<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 4, lev 5 fmwt apply left transpose - method 2")
    {
      relaxed_comparison(product_left_trans1, product_left_trans2);
    }

    auto const product_right1 = mat1 * fmwt;
    auto const product_right2 =
        apply_right_fmwt<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 4, lev 5 fmwt apply right - method 2")
    {
      relaxed_comparison(product_right1, product_right2);
    }

    auto const product_right_trans1 = mat1 * fmwt_transpose;
    auto const product_right_trans2 =
        apply_right_fmwt_transposed<TestType>(fmwt, mat1, kdeg, lev);
    SECTION("degree = 4, lev 5 fmwt apply right transpose - method 2")
    {
      relaxed_comparison(product_right_trans1, product_right_trans2);
    }
  }
}
