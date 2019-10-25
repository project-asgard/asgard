#include "distribution.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "transformations.hpp"
#include <numeric>

auto const tol_scale = 1e3;
TEMPLATE_TEST_CASE("combine dimensions", "[transformations]", double, float)
{
  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 1 rank")
  {
    int const dims = 2;
    int const lev  = 3;
    int const deg  = 2;

    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + "_sg.dat";

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

    int const num_ranks          = 1;
    distribution_plan const plan = get_plan(num_ranks, t);
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(filename));
    for (auto const &[rank, grid] : plan)
    {
      int const rank_start =
          grid.row_start * static_cast<int>(std::pow(deg, dims));
      int const rank_stop =
          (grid.row_stop + 1) * static_cast<int>(std::pow(deg, dims)) - 1;
      fk::vector<TestType, mem_type::view> const gold_partial(gold, rank_start,
                                                              rank_stop);
      REQUIRE(combine_dimensions(deg, t, plan.at(rank).row_start,
                                 plan.at(rank).row_stop, vectors,
                                 time) == gold_partial);
    }
  }

  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 8 ranks")
  {
    int const dims = 2;
    int const lev  = 3;
    int const deg  = 2;

    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + "_sg.dat";

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

    int const num_ranks          = 8;
    distribution_plan const plan = get_plan(num_ranks, t);
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(filename));
    fk::vector<TestType> test(gold.size());
    for (auto const &[rank, grid] : plan)
    {
      int const rank_start =
          grid.row_start * static_cast<int>(std::pow(deg, dims));
      int const rank_stop =
          (grid.row_stop + 1) * static_cast<int>(std::pow(deg, dims)) - 1;
      fk::vector<TestType, mem_type::view> const gold_partial(gold, rank_start,
                                                              rank_stop);
      fk::vector<TestType> const test_partial =
          combine_dimensions(deg, t, plan.at(rank).row_start,
                             plan.at(rank).row_stop, vectors, time);
      REQUIRE(test_partial == gold_partial);
      test.set_subvector(rank_start, test_partial);
    }
    REQUIRE(test == gold);
  }

  SECTION("combine dimensions, dim = 3, deg = 3, lev = 2, full grid, 20 ranks")
  {
    int const dims = 3;
    int const lev  = 2;
    int const deg  = 3;
    std::string const filename =
        "../testing/generated-inputs/transformations/combine_dim_dim" +
        std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
        std::to_string(lev) + "_fg.dat";

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

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(filename));

    fk::vector<TestType> test(gold.size());
    int const num_ranks = 20;
    auto const plan     = get_plan(num_ranks, t);
    for (auto const &[rank, grid] : plan)
    {
      int const rank_start =
          grid.row_start * static_cast<int>(std::pow(deg, dims));
      int const rank_stop =
          (grid.row_stop + 1) * static_cast<int>(std::pow(deg, dims)) - 1;
      fk::vector<TestType, mem_type::view> const gold_partial(gold, rank_start,
                                                              rank_stop);
      fk::vector<TestType> const test_partial =
          combine_dimensions(deg, t, plan.at(rank).row_start,
                             plan.at(rank).row_stop, vectors, time);
      REQUIRE(test_partial == gold_partial);
      test.set_subvector(rank_start, test_partial);
    }
    REQUIRE(test == gold);
  }
}

TEMPLATE_TEST_CASE("forward multi-wavelet transform", "[transformations]",
                   double, float)
{
  SECTION("transform(2, 2, -1, 1, double)")
  {
    int const degree     = 2;
    int const levels     = 2;
    auto const double_it = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
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
    relaxed_comparison(gold, test, tol_scale);
  }

  SECTION("transform(3, 4, -2.0, 2.0, double plus)")
  {
    int const degree       = 3;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
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

    relaxed_comparison(gold, test, tol_scale);
  }
}

TEMPLATE_TEST_CASE("wavelet_to_realspace", "[transformations]", double, float)
{
  auto const tol_factor = std::is_same<float, TestType>::value ? 1e5 : 1e8;

  /* memory limit for routines */
  int const limit_MB = 4000;

  SECTION("wavelet_to_realspace_1")
  {
    int const level  = 8;
    int const degree = 7;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);
    std::vector<TestType> wave_space;
    int const wave_space_size = table.size() * std::pow(degree, pde->num_dims);

    /* arbitrary function to transform from wavelet space to real space */
    auto const arbitrary_func = [](TestType x) -> TestType { return 2.0 * x; };

    /* simple wave space function to transform */
    for (int i = 0; i < wave_space_size; i++)
    {
      wave_space.push_back(arbitrary_func(i));
    }

    fk::vector<TestType> const realspace = wavelet_to_realspace<TestType>(
        *pde, fk::vector<TestType>(wave_space), table, limit_MB);
    std::string const gold_file_name =
        "../testing/generated-inputs/transformations/wavelet_to_realspace/"
        "continuity_1/"
        "wavelet_to_realspace.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(gold_file_name));
    relaxed_comparison(gold, realspace, tol_factor);
  }

  SECTION("wavelet_to_realspace_2")
  {
    int const level  = 4;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);
    std::vector<TestType> wave_space;
    int const wave_space_size = table.size() * std::pow(degree, pde->num_dims);

    /* arbitrary function to transform from wavelet space to real space */
    auto const arbitrary_func = [](TestType x) -> TestType { return 2.0 * x; };

    /* simple wave space function to transform */
    for (int i = 0; i < wave_space_size; i++)
    {
      wave_space.push_back(arbitrary_func(i));
    }

    fk::vector<TestType> const realspace = wavelet_to_realspace<TestType>(
        *pde, fk::vector<TestType>(wave_space), table, limit_MB);
    std::string const gold_file_name =
        "../testing/generated-inputs/transformations/wavelet_to_realspace/"
        "continuity_2/"
        "wavelet_to_realspace.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(gold_file_name));
    relaxed_comparison(gold, realspace, tol_factor);
  }

  SECTION("wavelet_to_realspace_3")
  {
    int const level  = 3;
    int const degree = 4;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);
    std::vector<TestType> wave_space;
    int const wave_space_size = table.size() * std::pow(degree, pde->num_dims);

    /* arbitrary function to transform from wavelet space to real space */
    auto const arbitrary_func = [](TestType x) -> TestType { return 2.0 * x; };

    /* simple wave space function to transform */
    for (int i = 0; i < wave_space_size; i++)
    {
      wave_space.push_back(arbitrary_func(i));
    }

    fk::vector<TestType> const realspace = wavelet_to_realspace<TestType>(
        *pde, fk::vector<TestType>(wave_space), table, limit_MB);
    std::string const gold_file_name =
        "../testing/generated-inputs/transformations/wavelet_to_realspace/"
        "continuity_3/"
        "wavelet_to_realspace.dat";
    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(gold_file_name));
    relaxed_comparison(gold, realspace, tol_factor);
  }
}

TEMPLATE_TEST_CASE("gen_realspace_transform", "[transformations]", double,
                   float)
{
  auto const tol_factor = std::is_same<float, TestType>::value ? 1e5 : 1e6;

  /* continuity_1 */
  SECTION("gen_realspace_transform_1")
  {
    /* Defaults to match gold */
    int const level  = 8;
    int const degree = 7;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    std::vector<fk::matrix<TestType>> realspace_transform =
        gen_realspace_transform(*pde);

    /* check against gold data */
    for (int i = 0; i < static_cast<int>(realspace_transform.size()); i++)
    {
      std::string const gold_file_name =
          "../testing/generated-inputs/transformations/matrix_plot_D/"
          "continuity_1/"
          "matrix_plot_D_" +
          std::to_string(i) + ".dat";
      fk::matrix<TestType> const gold =
          fk::matrix<TestType>(read_matrix_from_txt_file(gold_file_name));
      relaxed_comparison(gold, realspace_transform[i], tol_factor);
    }
  }

  /* continuity_2 */
  SECTION("gen_realspace_transform_2")
  {
    /* Defaults to match gold */
    int const level  = 7;
    int const degree = 6;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    std::vector<fk::matrix<TestType>> realspace_transform =
        gen_realspace_transform(*pde);

    /* check against gold data */
    for (int i = 0; i < static_cast<int>(realspace_transform.size()); i++)
    {
      std::string const gold_file_name =
          "../testing/generated-inputs/transformations/matrix_plot_D/"
          "continuity_2/"
          "matrix_plot_D_" +
          std::to_string(i) + ".dat";
      fk::matrix<TestType> const gold =
          fk::matrix<TestType>(read_matrix_from_txt_file(gold_file_name));
      relaxed_comparison(gold, realspace_transform[i], tol_factor);
    }
  }

  /* continuity_3 */
  SECTION("gen_realspace_transform_3")
  {
    /* Defaults to match gold */
    int const level  = 6;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    std::vector<fk::matrix<TestType>> realspace_transform =
        gen_realspace_transform(*pde);

    /* check against gold data */
    for (int i = 0; i < static_cast<int>(realspace_transform.size()); i++)
    {
      std::string const gold_file_name =
          "../testing/generated-inputs/transformations/matrix_plot_D/"
          "continuity_3/"
          "matrix_plot_D_" +
          std::to_string(i) + ".dat";
      fk::matrix<TestType> const gold =
          fk::matrix<TestType>(read_matrix_from_txt_file(gold_file_name));
      relaxed_comparison(gold, realspace_transform[i], tol_factor);
    }
  }

  /* continuity_6 */
  SECTION("gen_realspace_transform_6")
  {
    /* Defaults to match gold */
    int const level  = 2;
    int const degree = 3;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);
    std::vector<fk::matrix<TestType>> realspace_transform =
        gen_realspace_transform(*pde);

    /* check against gold data */
    for (int i = 0; i < static_cast<int>(realspace_transform.size()); i++)
    {
      std::string const gold_file_name =
          "../testing/generated-inputs/transformations/matrix_plot_D/"
          "continuity_6/"
          "matrix_plot_D_" +
          std::to_string(i) + ".dat";
      fk::matrix<TestType> const gold =
          fk::matrix<TestType>(read_matrix_from_txt_file(gold_file_name));
      relaxed_comparison(gold, realspace_transform[i], tol_factor);
    }
  }
}
