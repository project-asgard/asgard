
#include "distribution.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tests_general.hpp"
#include "tools.hpp"
#include "transformations.hpp"
#include <climits>
#include <numeric>

template<typename P>
void test_combine_dimensions(PDE<P> const &pde, P const time = 1.0,
                             int const num_ranks  = 1,
                             bool const full_grid = false)
{
  int const dims = pde.num_dims;

  // FIXME assuming uniform degree across dims
  dimension const dim = pde.get_dimensions()[0];
  int const lev       = dim.get_level();
  int const deg       = dim.get_degree();

  std::string const filename =
      "../testing/generated-inputs/transformations/combine_dim_dim" +
      std::to_string(dims) + "_deg" + std::to_string(deg) + "_lev" +
      std::to_string(lev) + "_" + (full_grid ? "fg" : "sg") + ".dat";

  std::string const grid_str = full_grid ? "-f" : "";
  options const o            = make_options(
      {"-d", std::to_string(deg), "-l", std::to_string(lev), grid_str});

  elements::table const t(o, pde);

  std::vector<fk::vector<P>> vectors;
  P counter = 1.0;
  for (int i = 0; i < pde.num_dims; ++i)
  {
    int const vect_size         = dims * static_cast<int>(std::pow(2, lev));
    fk::vector<P> const vect_1d = [&counter, vect_size] {
      fk::vector<P> vect(vect_size);
      std::iota(vect.begin(), vect.end(), static_cast<P>(counter));
      counter += vect.size();
      return vect;
    }();
    vectors.push_back(vect_1d);
  }
  distribution_plan const plan = get_plan(num_ranks, t);

  fk::vector<P> const gold = fk::vector<P>(read_vector_from_txt_file(filename));
  fk::vector<P> test(gold.size());
  for (auto const &[rank, grid] : plan)
  {
    int const rank_start =
        grid.row_start * static_cast<int>(std::pow(deg, dims));
    int const rank_stop =
        (grid.row_stop + 1) * static_cast<int>(std::pow(deg, dims)) - 1;
    fk::vector<P, mem_type::const_view> const gold_partial(gold, rank_start,
                                                           rank_stop);
    fk::vector<P> const test_partial = combine_dimensions(
        deg, t, plan.at(rank).row_start, plan.at(rank).row_stop, vectors, time);
    REQUIRE(test_partial == gold_partial);
    test.set_subvector(rank_start, test_partial);
  }
  REQUIRE(test == gold);
}

TEMPLATE_TEST_CASE("combine dimensions", "[transformations]", double, float)
{
  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 1 rank")
  {
    int const lev       = 3;
    int const deg       = 2;
    auto const pde      = make_PDE<TestType>(PDE_opts::continuity_2, lev, deg);
    TestType const time = 2.0;
    test_combine_dimensions(*pde, time);
  }

  SECTION("combine dimensions, dim = 2, deg = 2, lev = 3, 8 ranks")
  {
    int const lev       = 3;
    int const deg       = 2;
    auto const pde      = make_PDE<TestType>(PDE_opts::continuity_2, lev, deg);
    int const num_ranks = 8;
    TestType const time = 2.0;
    test_combine_dimensions(*pde, time, num_ranks);
  }

  SECTION("combine dimensions, dim = 3, deg = 3, lev = 2, full grid")
  {
    int const lev        = 2;
    int const deg        = 3;
    auto const pde       = make_PDE<TestType>(PDE_opts::continuity_3, lev, deg);
    int const num_ranks  = 20;
    TestType const time  = 2.5;
    bool const full_grid = true;
    test_combine_dimensions(*pde, time, num_ranks, full_grid);
  }
}

TEMPLATE_TEST_CASE("forward multi-wavelet transform", "[transformations]",
                   double, float)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-16 : 1e-6;

  SECTION("transform(2, 2, -1, 1, double)")
  {
    int const degree     = 2;
    int const levels     = 2;
    auto const double_it = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x * static_cast<TestType>(2.0);
    };

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, levels, degree);
    auto const dim = pde->get_dimensions()[0];

    auto const opts = make_options(
        {"-l", std::to_string(levels), "-d", std::to_string(degree)});

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);

    auto const gold = fk::vector<TestType>(read_vector_from_txt_file(
        "../testing/generated-inputs/transformations/forward_transform_" +
        std::to_string(degree) + "_" + std::to_string(levels) +
        "_neg1_pos1_double.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_it, transformer);

    rmse_comparison(gold, test, tol_factor);
  }

  SECTION("transform(3, 4, -2.0, 2.0, double plus)")
  {
    int const degree       = 3;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x + (x * static_cast<TestType>(2.0));
    };

    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, levels, degree);
    auto const dim = pde->get_dimensions()[1];

    auto const opts = make_options(
        {"-l", std::to_string(levels), "-d", std::to_string(degree)});

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);

    fk::vector<TestType> const gold =
        fk::vector<TestType>(read_vector_from_txt_file(
            "../testing/generated-inputs/transformations/forward_transform_" +
            std::to_string(degree) + "_" + std::to_string(levels) +
            "_neg2_pos2_doubleplus.dat"));
    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_plus, transformer);

    rmse_comparison(gold, test, tol_factor);
  }
}

template<typename P>
void test_wavelet_to_realspace(PDE<P> const &pde,
                               std::string const &gold_filename,
                               P const tol_factor)
{
  // memory limit for routines
  static auto constexpr limit_MB = 4000;

  // FIXME assume uniform level and degree
  auto const &d     = pde.get_dimensions()[0];
  auto const level  = d.get_level();
  auto const degree = d.get_degree();

  auto const opts = make_options({"-l", std::to_string(level)});
  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);
  elements::table const table(opts, pde);

  auto const wave_space = [&table, &pde, degree]() {
    // arbitrary function to transform from wavelet space to real space
    auto const arbitrary_func = [](P const x) { return 2.0 * x; };

    auto const wave_space_size =
        static_cast<uint64_t>(table.size()) * std::pow(degree, pde.num_dims);
    tools::expect(wave_space_size < INT_MAX);
    fk::vector<P> wave_space(wave_space_size);

    for (int i = 0; i < wave_space.size(); ++i)
    {
      wave_space(i) = arbitrary_func(i);
    }
    return wave_space;
  }();

  auto const real_space_size = real_solution_size(pde);
  fk::vector<P> real_space(real_space_size);

  fk::vector<P, mem_type::owner, resource::host> workspace_0(real_space_size);
  fk::vector<P, mem_type::owner, resource::host> workspace_1(real_space_size);

  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace_0),
      fk::vector<P, mem_type::view, resource::host>(workspace_1)};

  wavelet_to_realspace<P>(pde, wave_space, table, transformer, limit_MB,
                          tmp_workspace, real_space);

  auto const gold = fk::vector<P>(read_vector_from_txt_file(gold_filename));

  rmse_comparison(gold, real_space, tol_factor);
}

TEMPLATE_TEST_CASE("wavelet_to_realspace", "[transformations]", double, float)
{
  SECTION("wavelet_to_realspace_1")
  {
    int const level  = 8;
    int const degree = 7;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/"
        "wavelet_to_realspace_continuity_1.dat";

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-11 : 1e-2;
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }

  SECTION("wavelet_to_realspace_2")
  {
    int const level  = 4;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/"
        "wavelet_to_realspace_continuity_2.dat";

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-13 : 1e-4;
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }

  SECTION("wavelet_to_realspace_3")
  {
    int const level  = 3;
    int const degree = 4;
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/"
        "wavelet_to_realspace_continuity_3.dat";

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-14 : 1e-5;
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }
}

template<typename P>
void test_gen_realspace_transform(PDE<P> const &pde,
                                  std::string const &gold_filename,
                                  P const tol_factor)
{
  // FIXME assumes uniform level and degree across dims
  auto const level  = pde.get_dimensions()[0].get_level();
  auto const degree = pde.get_dimensions()[0].get_degree();
  auto const opts =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});
  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);
  std::vector<fk::matrix<P>> const transforms =
      gen_realspace_transform(pde, transformer);

  for (int i = 0; i < static_cast<int>(transforms.size()); ++i)
  {
    fk::matrix<P> const gold = fk::matrix<P>(
        read_matrix_from_txt_file(gold_filename + std::to_string(i) + ".dat"));
    rmse_comparison(gold, transforms[i], tol_factor);
  }
}

TEMPLATE_TEST_CASE("gen_realspace_transform", "[transformations]", double,
                   float)
{
  SECTION("gen_realspace_transform_1")
  {
    int const level  = 7;
    int const degree = 7;
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/matrix_plot_D/"
        "continuity_1/"
        "matrix_plot_D_";
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_1, level, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-12 : 1e-2;
    test_gen_realspace_transform(*pde, gold_filename, tol_factor);
  }

  SECTION("gen_realspace_transform_2")
  {
    int const level  = 7;
    int const degree = 6;
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/matrix_plot_D/"
        "continuity_2/"
        "matrix_plot_D_";
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-12 : 1e-4;
    test_gen_realspace_transform(*pde, gold_filename, tol_factor);
  }

  SECTION("gen_realspace_transform_3")
  {
    int const level  = 6;
    int const degree = 5;
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/matrix_plot_D/"
        "continuity_3/"
        "matrix_plot_D_";
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-14 : 1e-4;
    test_gen_realspace_transform(*pde, gold_filename, tol_factor);
  }

  SECTION("gen_realspace_transform_6")
  {
    int const level  = 2;
    int const degree = 3;
    std::string const gold_filename =
        "../testing/generated-inputs/transformations/matrix_plot_D/"
        "continuity_6/"
        "matrix_plot_D_";
    auto const pde = make_PDE<TestType>(PDE_opts::continuity_6, level, degree);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-14 : 1e-6;
    test_gen_realspace_transform(*pde, gold_filename, tol_factor);
  }
}
