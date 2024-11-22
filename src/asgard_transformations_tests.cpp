#include "tests_general.hpp"

#include "asgard_small_mats.hpp"

static auto const transformations_base_dir = gold_base_dir / "transformations";

using namespace asgard;

template<typename P>
void test_combine_dimensions(PDE<P> const &pde, P const time = 1.0,
                             int const num_ranks  = 1,
                             bool const full_grid = false)
{
  int const dims = pde.num_dims();

  dimension const dim = pde.get_dimensions()[0];
  int const lev       = dim.get_level();
  int const degree    = dim.get_degree();

  std::string const filename =
      "combine_dim_dim" + std::to_string(dims) + "_deg" + std::to_string(degree + 1) +
      "_lev" + std::to_string(lev) + "_" + (full_grid ? "fg" : "sg") + ".dat";

  elements::table const t(pde);

  std::vector<fk::vector<P>> vectors;
  P counter = 1.0;
  for (int i = 0; i < pde.num_dims(); ++i)
  {
    int const vect_size         = dims * fm::ipow2(lev);
    fk::vector<P> const vect_1d = [&counter, vect_size] {
      fk::vector<P> vect(vect_size);
      std::iota(vect.begin(), vect.end(), static_cast<P>(counter));
      counter += vect.size();
      return vect;
    }();
    vectors.push_back(vect_1d);
  }
  distribution_plan const plan = get_plan(num_ranks, t);

  fk::vector<P> const gold =
      read_vector_from_txt_file<P>(transformations_base_dir / filename);
  fk::vector<P> test(gold.size());
  for (auto const &[rank, grid] : plan)
  {
    int const rank_start =
        grid.row_start * fm::ipow(degree + 1, dims);
    int const rank_stop =
        (grid.row_stop + 1) * fm::ipow(degree + 1, dims) - 1;
    fk::vector<P, mem_type::const_view> const gold_partial(gold, rank_start,
                                                           rank_stop);
    fk::vector<P> const test_partial = combine_dimensions(
        degree, t, plan.at(rank).row_start, plan.at(rank).row_stop, vectors, time);
    REQUIRE(test_partial == gold_partial);
    test.set_subvector(rank_start, test_partial);
  }
  REQUIRE(test == gold);
}

TEMPLATE_TEST_CASE("combine dimensions", "[transformations]", test_precs)
{
  SECTION("combine dimensions, dim = 2, degree = 2, lev = 3, 1 rank")
  {
    auto const pde = make_PDE<TestType>("-p continuity_2 -l 3 -d 1");

    TestType const time = 2.0;
    test_combine_dimensions(*pde, time);
  }

  SECTION("combine dimensions, dim = 2, degree = 1, lev = 3, 8 ranks")
  {
    auto const pde = make_PDE<TestType>("-p continuity_2 -l 3 -d 1");
    int const num_ranks = 8;
    TestType const time = 2.0;
    test_combine_dimensions(*pde, time, num_ranks);
  }

  SECTION("combine dimensions, dim = 3, degree = 2, lev = 2, full grid")
  {
    auto const pde = make_PDE<TestType>("-p continuity_3 -l 2 -d 2 -g dense");
    int const num_ranks  = 20;
    TestType const time  = 2.5;
    bool const full_grid = true;
    test_combine_dimensions(*pde, time, num_ranks, full_grid);
  }
}

TEMPLATE_TEST_CASE("forward multi-wavelet transform", "[transformations]",
                   test_precs)
{
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("transform(1, 2, -1, 1, double)")
  {
    int const degree     = 1;
    int const levels     = 2;
    auto const double_it = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x * static_cast<TestType>(2.0);
    };
    g_func_type<TestType> const tenth_func = [](TestType x, TestType t) {
      ignore(t);
      return x * 0.1;
    };

    auto const pde = make_PDE<TestType>("-p continuity_1 -l 2 -d 1");
    auto const dim = pde->get_dimensions()[0];

    basis::wavelet_transform<TestType, resource::host> const transformer(*pde);

    auto const gold = read_vector_from_txt_file<TestType>(
        transformations_base_dir /
        ("forward_transform_" + std::to_string(degree + 1) + "_" +
         std::to_string(levels) + "_neg1_pos1_double.dat"));

    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_it, tenth_func, transformer);

    rmse_comparison(gold, test, tol_factor);
  }

  SECTION("transform(2, 4, -2.0, 2.0, double plus)")
  {
    int const degree       = 2;
    int const levels       = 4;
    auto const double_plus = [](fk::vector<TestType> x, TestType t) {
      ignore(t);
      return x + (x * static_cast<TestType>(2.0));
    };
    g_func_type<TestType> const tenth_func = [](TestType x, TestType t) {
      ignore(t);
      return x * 0.1;
    };

    auto const pde = make_PDE<TestType>("-p continuity_2 -l 4 -d 2");
    auto const dim = pde->get_dimensions()[1];

    basis::wavelet_transform<TestType, resource::host> const transformer(*pde);

    fk::vector<TestType> const gold = read_vector_from_txt_file<TestType>(
        transformations_base_dir /
        ("forward_transform_" + std::to_string(degree + 1) + "_" +
         std::to_string(levels) + "_neg2_pos2_doubleplus.dat"));
    fk::vector<TestType> const test =
        forward_transform<TestType>(dim, double_plus, tenth_func, transformer);

    rmse_comparison(gold, test, tol_factor);
  }
}

template<typename P>
void test_wavelet_to_realspace(PDE<P> const &pde,
                               std::string const &gold_filename,
                               P const tol_factor)
{
  auto const degree = pde.options().degree.value();

  basis::wavelet_transform<P, resource::host> const transformer(pde);
  elements::table const table(pde);

  auto const wave_space = [&table, &pde, degree]() {
    // arbitrary function to transform from wavelet space to real space
    auto const arbitrary_func = [](P const x) { return 2.0 * x; };

    auto const wave_space_size =
        static_cast<uint64_t>(table.size()) * fm::ipow(degree + 1, pde.num_dims());
    expect(wave_space_size < INT_MAX);
    fk::vector<P> wave_space_in(wave_space_size);

    for (int i = 0; i < wave_space_in.size(); ++i)
    {
      wave_space_in(i) = arbitrary_func(i);
    }
    return wave_space_in;
  }();

  auto const dense_size = dense_space_size(pde);
  fk::vector<P> real_space(dense_size);

  fk::vector<P, mem_type::owner, resource::host> workspace_0(dense_size);
  fk::vector<P, mem_type::owner, resource::host> workspace_1(dense_size);

  std::array<fk::vector<P, mem_type::view, resource::host>, 2> tmp_workspace = {
      fk::vector<P, mem_type::view, resource::host>(workspace_0),
      fk::vector<P, mem_type::view, resource::host>(workspace_1)};

  wavelet_to_realspace<P>(pde, wave_space, table, transformer, tmp_workspace,
                          real_space, quadrature_mode::use_degree);

  auto const gold = read_vector_from_txt_file<P>(gold_filename);

  rmse_comparison(gold, real_space, tol_factor);
}

TEMPLATE_TEST_CASE("wavelet_to_realspace", "[transformations]", test_precs)
{
  SECTION("wavelet_to_realspace_1")
  {
    auto const pde = make_PDE<TestType>("-p continuity_1 -l 8 -d 6");
    auto const gold_filename =
        transformations_base_dir / "wavelet_to_realspace_continuity_1.dat";

    auto constexpr tol_factor = get_tolerance<TestType>(100000);
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }

  SECTION("wavelet_to_realspace_2")
  {
    auto const pde = make_PDE<TestType>("-p continuity_2 -l 4 -d 4");
    auto const gold_filename =
        transformations_base_dir / "wavelet_to_realspace_continuity_2.dat";

    auto constexpr tol_factor = get_tolerance<TestType>(100000);
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }

  SECTION("wavelet_to_realspace_3")
  {
    auto const pde = make_PDE<TestType>("-p continuity_3 -l 3 -d 3");
    auto const gold_filename =
        transformations_base_dir / "wavelet_to_realspace_continuity_3.dat";

    auto constexpr tol_factor = get_tolerance<TestType>(10);
    test_wavelet_to_realspace(*pde, gold_filename, tol_factor);
  }
}

template<typename P>
void test_gen_realspace_transform(PDE<P> const &pde,
                                  std::filesystem::path const &gold_directory,
                                  std::string const &gold_filename,
                                  P const tol_factor)
{
  basis::wavelet_transform<P, resource::host> const transformer(pde);
  std::vector<fk::matrix<P>> const transforms =
      gen_realspace_transform(pde, transformer, quadrature_mode::use_degree);

  for (int i = 0; i < static_cast<int>(transforms.size()); ++i)
  {
    fk::matrix<P> const gold = read_matrix_from_txt_file<P>(
        gold_directory / (gold_filename + std::to_string(i) + ".dat"));
    rmse_comparison(gold, transforms[i], tol_factor);
  }
}

TEMPLATE_TEST_CASE("gen_realspace_transform", "[transformations]", test_precs)
{
  SECTION("gen_realspace_transform_1")
  {
    auto const gold_directory =
        transformations_base_dir / "matrix_plot_D/continuity_1";
    std::string const gold_filename = "matrix_plot_D_";

    auto const pde = make_PDE<TestType>("-p continuity_1 -l 7 -d 6");

    auto constexpr tol_factor = get_tolerance<TestType>(10000);
    test_gen_realspace_transform(*pde, gold_directory, gold_filename,
                                 tol_factor);
  }

  SECTION("gen_realspace_transform_2")
  {
    auto const gold_directory =
        transformations_base_dir / "matrix_plot_D/continuity_2";
    std::string const gold_filename = "matrix_plot_D_";

    auto const pde = make_PDE<TestType>("-p continuity_2 -l 7 -d 5");
    auto constexpr tol_factor = get_tolerance<TestType>(1000);
    test_gen_realspace_transform(*pde, gold_directory, gold_filename,
                                 tol_factor);
  }

  SECTION("gen_realspace_transform_3")
  {
    auto const gold_directory =
        transformations_base_dir / "matrix_plot_D/continuity_3/";
    std::string const gold_filename = "matrix_plot_D_";

    auto const pde = make_PDE<TestType>("-p continuity_3 -l 6 -d 4");
    auto constexpr tol_factor = get_tolerance<TestType>(100);
    test_gen_realspace_transform(*pde, gold_directory, gold_filename,
                                 tol_factor);
  }

  SECTION("gen_realspace_transform_6")
  {
    auto const gold_directory =
        transformations_base_dir / "matrix_plot_D/continuity_6";
    std::string const gold_filename = "matrix_plot_D_";

    auto const pde = make_PDE<TestType>("-p continuity_6 -l 2 -d 2");

    auto constexpr tol_factor = get_tolerance<TestType>(20);
    test_gen_realspace_transform(*pde, gold_directory, gold_filename,
                                 tol_factor);
  }
}

