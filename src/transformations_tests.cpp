#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 250) ==
              second_elem);
    });
  };

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

    TestType h0 = static_cast<TestType>(readScalarFromTxtFile(h0_string));
    TestType h1 = static_cast<TestType>(readScalarFromTxtFile(h1_string));
    TestType g0 = static_cast<TestType>(readScalarFromTxtFile(g0_string));
    TestType g1 = static_cast<TestType>(readScalarFromTxtFile(g1_string));
    fk::matrix<TestType> phi_co = readMatrixFromTxtFile(phi_string);
    TestType scale_co =
        static_cast<TestType>(readScalarFromTxtFile(scale_string));

    auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
        generate_multi_wavelets<TestType>(degree);

    SECTION("degree = 1, h0") { REQUIRE(Approx(h0) == m_h0(0, 0)); }

    SECTION("degree = 1, h1") { REQUIRE(Approx(h1) == m_h1(0, 0)); }
    SECTION("degree = 1, g0") { REQUIRE(Approx(g0) == m_g0(0, 0)); }
    SECTION("degree = 1, g1") { REQUIRE(Approx(g1) == m_g1(0, 0)); }
    SECTION("degree = 1, phi_co") { relaxed_comparison(phi_co, m_phi_co); }
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

    fk::matrix<TestType> h0       = readMatrixFromTxtFile(h0_string);
    fk::matrix<TestType> h1       = readMatrixFromTxtFile(h1_string);
    fk::matrix<TestType> g0       = readMatrixFromTxtFile(g0_string);
    fk::matrix<TestType> g1       = readMatrixFromTxtFile(g1_string);
    fk::matrix<TestType> phi_co   = readMatrixFromTxtFile(phi_string);
    fk::matrix<TestType> scale_co = readMatrixFromTxtFile(scale_string);

    auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
        generate_multi_wavelets<TestType>(degree);

    SECTION("degree = 3, h0") { relaxed_comparison(h0, m_h0); }
    SECTION("degree = 3, h1") { relaxed_comparison(h1, m_h1); }
    SECTION("degree = 3, g0") { relaxed_comparison(g0, m_g0); }
    SECTION("degree = 3, g1") { relaxed_comparison(g1, m_g1); }
    SECTION("degree = 3, phi_co") { relaxed_comparison(phi_co, m_phi_co); }
    SECTION("degree = 3, scale_co")
    {
      relaxed_comparison(scale_co, m_scale_co);
    }
  }
}

TEMPLATE_TEST_CASE("Combine dimensions", "[transformations]", double, float)
{
  SECTION("Combine dimensions, dim = 2, deg = 2, lev = 3")
  {
    int const dim = 2;
    int const lev = 3;
    int const deg = 2;
    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dim) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + ".dat";

    fk::vector<TestType> const gold = readVectorFromTxtFile(filename);

    Options const o =
        make_options({"-d", std::to_string(deg), "-l", std::to_string(lev)});
    element_table const t(o, dim);

    TestType const time = 2.0;

    int const vect_size              = dim * static_cast<int>(std::pow(2, lev));
    fk::vector<TestType> const dim_1 = [&] {
      fk::vector<TestType> dim_1(vect_size);
      std::iota(dim_1.begin(), dim_1.end(), static_cast<TestType>(1.0));
      return dim_1;
    }();
    fk::vector<TestType> const dim_2 = [&] {
      fk::vector<TestType> dim_2(vect_size);
      std::iota(dim_2.begin(), dim_2.end(),
                dim_1(dim_1.size() - 1) + static_cast<TestType>(1.0));
      return dim_2;
    }();
    std::vector<fk::vector<TestType>> const vectors = {dim_1, dim_2};

    REQUIRE(combine_dimensions(o, t, vectors, time) == gold);
  }
}
