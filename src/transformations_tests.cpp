#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"

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

    TestType h0 = static_cast<TestType>(readScalarFromTxtFile(h0_string));
    TestType h1 = static_cast<TestType>(readScalarFromTxtFile(h1_string));
    TestType g0 = static_cast<TestType>(readScalarFromTxtFile(g0_string));
    TestType g1 = static_cast<TestType>(readScalarFromTxtFile(g1_string));
    fk::matrix<TestType> phi_co = readMatrixFromTxtFile(phi_string);
    TestType scale_co =
        static_cast<TestType>(readScalarFromTxtFile(scale_string));

    auto const [m_h0, m_h1, m_g0, m_g1, m_phi_co, m_scale_co] =
        generate_multi_wavelets<TestType>(degree);

    REQUIRE(Approx(h0) == m_h0(0, 0));
    REQUIRE(Approx(h1) == m_h1(0, 0));
    REQUIRE(Approx(g0) == m_g0(0, 0));
    REQUIRE(Approx(g1) == m_g1(0, 0));
    relaxed_comparison(phi_co, m_phi_co);
    REQUIRE(Approx(scale_co) == m_scale_co(0, 0));
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

    relaxed_comparison(h0, m_h0);
    REQUIRE(h1 == m_h1);
    REQUIRE(g0 == m_g0);
    REQUIRE(g1 == m_g1);
    REQUIRE(phi_co == m_phi_co);
    REQUIRE(scale_co == m_scale_co);
  }
}
