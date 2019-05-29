#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>

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

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(filename));
    dimension const dim = make_PDE<TestType>(PDE_opts::continuity_1, lev, deg)
                              ->get_dimensions()[0];

    options const o =
        make_options({"-d", std::to_string(deg), "-l", std::to_string(lev)});
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

    REQUIRE(combine_dimensions(deg, t, vectors, time) == gold);
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

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(filename));

    dimension const dim = make_PDE<TestType>(PDE_opts::continuity_1, lev, deg)
                              ->get_dimensions()[0];
    options const o = make_options(
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

    REQUIRE(combine_dimensions(deg, t, vectors, time) == gold);
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

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_1, levels, degree)
            ->get_dimensions()[0];
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(
            "../testing/generated-inputs/transformations/forward_transform_" +
            std::to_string(degree) + "_" + std::to_string(levels) +
            "_neg1_pos1_double.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_it);
    relaxed_comparison(gold, test);
  }

  SECTION("transform(3, 4, -2.0, 2.0, double plus)")
  {
    int const degree       = 3;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x) {
      return x + (x * static_cast<TestType>(2.0));
    };

    dimension const dim =
        make_PDE<TestType>(PDE_opts::continuity_2, levels, degree)
            ->get_dimensions()[1];

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(
            "../testing/generated-inputs/transformations/forward_transform_" +
            std::to_string(degree) + "_" + std::to_string(levels) +
            "_neg2_pos2_doubleplus.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_plus);

    relaxed_comparison(gold, test);
  }
}
