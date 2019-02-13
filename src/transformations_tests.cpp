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
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e3) ==
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

    TestType h0 = static_cast<TestType>(read_scalar_from_txt_file(h0_string));
    TestType h1 = static_cast<TestType>(read_scalar_from_txt_file(h1_string));
    TestType g0 = static_cast<TestType>(read_scalar_from_txt_file(g0_string));
    TestType g1 = static_cast<TestType>(read_scalar_from_txt_file(g1_string));
    fk::matrix<TestType> phi_co = read_matrix_from_txt_file(phi_string);
    TestType scale_co =
        static_cast<TestType>(read_scalar_from_txt_file(scale_string));

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

    fk::matrix<TestType> h0       = read_matrix_from_txt_file(h0_string);
    fk::matrix<TestType> h1       = read_matrix_from_txt_file(h1_string);
    fk::matrix<TestType> g0       = read_matrix_from_txt_file(g0_string);
    fk::matrix<TestType> g1       = read_matrix_from_txt_file(g1_string);
    fk::matrix<TestType> phi_co   = read_matrix_from_txt_file(phi_string);
    fk::matrix<TestType> scale_co = read_matrix_from_txt_file(scale_string);

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
    int const dims = 2;
    int const lev  = 3;
    int const deg  = 2;

    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + "_sg.dat";

    fk::vector<TestType> const gold = read_vector_from_txt_file(filename);
    dimension const dim             = make_dummy_dim<TestType>(lev, deg);
    options const o                 = make_options(
        {"-d", std::to_string(deg), "-l", std::to_string(lev), "-f"});
    element_table const t(o, dims);
    TestType const time = 2.0;

    int const vect_size = dims * static_cast<int>(std::pow(2, lev));
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

    REQUIRE(combine_dimensions(dim, t, vectors, time) == gold);
  }

  SECTION("Combine dimensions, dim = 3, deg = 3, lev = 2, full grid")
  {
    int const dims = 3;
    int const lev  = 2;
    int const deg  = 3;
    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + "_fg.dat";

    fk::vector<TestType> const gold = read_vector_from_txt_file(filename);

    dimension const dim = make_dummy_dim<TestType>(lev, deg);
    options const o     = make_options(
        {"-d", std::to_string(deg), "-l", std::to_string(lev), "-f"});
    element_table const t(o, dims);
    TestType const time = 2.5;

    int const vect_size = dims * static_cast<int>(std::pow(2, lev));
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

    fk::vector<TestType> const dim_3 = [&] {
      fk::vector<TestType> dim_3(vect_size);
      std::iota(dim_3.begin(), dim_3.end(),
                dim_2(dim_2.size() - 1) + static_cast<TestType>(1.0));
      return dim_3;
    }();
    std::vector<fk::vector<TestType>> const vectors = {dim_1, dim_2, dim_3};

    REQUIRE(combine_dimensions(dim, t, vectors, time) == gold);
  }
}

// FIXME we are still off after 12 dec places or so in double prec.
// need to think about these precision issues some more
TEMPLATE_TEST_CASE("operator_two_scale function working appropriately",
                   "[transformations]", double)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e4) ==
              second_elem);
    });
  };

  SECTION("operator_two_scale(2, 2)")
  {
    int const degree = 2;
    int const levels = 2;

    dimension const dim = make_dummy_dim<TestType>(levels, degree);

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test = operator_two_scale<TestType>(dim);
    relaxed_comparison(gold, test);
  }

  SECTION("operator_two_scale(2, 3)")
  {
    int const degree = 2;
    int const levels = 3;

    dimension const dim = make_dummy_dim<TestType>(levels, degree);

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test = operator_two_scale<TestType>(dim);
    relaxed_comparison(gold, test);
  }
  SECTION("operator_two_scale(4, 3)")
  {
    int const degree = 4;
    int const levels = 3;

    dimension const dim = make_dummy_dim<TestType>(levels, degree);

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test = operator_two_scale<TestType>(dim);
    relaxed_comparison(gold, test);
  }
  SECTION("operator_two_scale(5, 5)")
  {
    int const degree = 5;
    int const levels = 5;

    dimension const dim = make_dummy_dim<TestType>(levels, degree);

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test = operator_two_scale<TestType>(dim);
    relaxed_comparison(gold, test);
  }

  SECTION("operator_two_scale(2, 6)")
  {
    int const degree = 2;
    int const levels = 6;

    dimension const dim = make_dummy_dim<TestType>(levels, degree);

    fk::matrix<TestType> const gold = read_matrix_from_txt_file(
        "../testing/generated-inputs/transformations/operator_two_scale_" +
        std::to_string(degree) + "_" + std::to_string(levels) + ".dat");
    fk::matrix<TestType> const test = operator_two_scale<TestType>(dim);
    relaxed_comparison(gold, test);
  }
}

TEMPLATE_TEST_CASE("forward multi-wavelet transform", "[transformations]",
                   double, float)
{
  auto const relaxed_comparison = [](auto const first, auto const second) {
    auto first_it = first.begin();
    std::for_each(second.begin(), second.end(), [&first_it](auto &second_elem) {
      REQUIRE(Approx(*first_it++)
                  .epsilon(std::numeric_limits<TestType>::epsilon() * 1e4) ==
              second_elem);
    });
  };

  SECTION("transform(2, 2, -1, 1, double)")
  {
    int const degree     = 2;
    int const levels     = 2;
    auto const double_it = [](fk::vector<TestType> x) {
      return x * static_cast<TestType>(2.0);
    };
    TestType const domain_min = static_cast<TestType>(-1.0);
    TestType const domain_max = static_cast<TestType>(1.0);

    dimension const dim =
        make_dummy_dim<TestType>(levels, degree, domain_min, domain_max);

    fk::vector<TestType> const gold = read_vector_from_txt_file(
        "../testing/generated-inputs/transformations/forward_transform_" +
        std::to_string(degree) + "_" + std::to_string(levels) +
        "_neg1_pos1_double.dat");

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_it);
    relaxed_comparison(gold, test);
  }

  SECTION("transform(3, 4, -2.5, 2.5, double plus)")
  {
    int const degree       = 3;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x) {
      return x + (x * static_cast<TestType>(2.0));
    };
    TestType const domain_min = static_cast<TestType>(-2.5);
    TestType const domain_max = static_cast<TestType>(2.5);

    dimension const dim =
        make_dummy_dim<TestType>(levels, degree, domain_min, domain_max);

    fk::vector<TestType> const gold = read_vector_from_txt_file(
        "../testing/generated-inputs/transformations/forward_transform_" +
        std::to_string(degree) + "_" + std::to_string(levels) +
        "_neg2_5_pos2_5_doubleplus.dat");

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_plus);

    relaxed_comparison(gold, test);
  }
}
