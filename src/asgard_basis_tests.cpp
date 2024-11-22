#include "tests_general.hpp"

#ifdef ASGARD_ENABLE_DOUBLE
#ifdef ASGARD_ENABLE_FLOAT

#ifdef ASGARD_USE_CUDA
#define mtest_precs                                     \
  (double, resource::host), (double, resource::device), \
      (float, resource::host), (float, resource::device)
#else
#define mtest_precs (double, resource::host), (float, resource::host)
#endif

#else

#ifdef ASGARD_USE_CUDA
#define mtest_precs (double, resource::host), (double, resource::device)
#else
#define mtest_precs (double, resource::host)
#endif

#endif
#else

#ifdef ASGARD_USE_CUDA
#define mtest_precs (float, resource::host), (float, resource::device)
#else
#define mtest_precs (float, resource::host)
#endif

#endif

static auto const transformations_base_dir = gold_base_dir / "transformations";

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

template<typename P>
void test_multiwavelet_gen(int const degree, P const tol_factor)
{
  std::string const out_base = "multiwavelet_" + std::to_string(degree + 1) + "_";

  auto const [m_h0, m_h1, m_g0, m_g1] = generate_multi_wavelets<P>(degree);

  auto const [h0, h1, g0, g1, scale_co] = [&out_base, degree]() {
    auto const h0_string = transformations_base_dir / (out_base + "h0.dat");
    auto const h1_string = transformations_base_dir / (out_base + "h1.dat");
    auto const g0_string = transformations_base_dir / (out_base + "g0.dat");
    auto const g1_string = transformations_base_dir / (out_base + "g1.dat");

    if (degree < 1)
    {
      auto const h0_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h0_string))}};
      auto const h1_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(h1_string))}};
      auto const g0_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g0_string))}};
      auto const g1_out =
          fk::matrix<P>{{static_cast<P>(read_scalar_from_txt_file(g1_string))}};
      return std::array<fk::matrix<P>, 5>{h0_out, h1_out, g0_out, g1_out};
    }
    else
    {
      fk::matrix<P> const h0_out = read_matrix_from_txt_file<P>(h0_string);
      fk::matrix<P> const h1_out = read_matrix_from_txt_file<P>(h1_string);
      fk::matrix<P> const g0_out = read_matrix_from_txt_file<P>(g0_string);
      fk::matrix<P> const g1_out = read_matrix_from_txt_file<P>(g1_string);
      return std::array<fk::matrix<P>, 5>{h0_out, h1_out, g0_out, g1_out};
    }
  }();

  rmse_comparison(h0, m_h0, tol_factor);
  rmse_comparison(h1, m_h1, tol_factor);
  rmse_comparison(g0, m_g0, tol_factor);
  rmse_comparison(g1, m_g1, tol_factor);
}

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", test_precs)
{
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  SECTION("Multiwavelet generation, degree = 0")
  {
    int const degree = 0;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 1")
  {
    int const degree = 1;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 2")
  {
    int const degree = 2;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }

  SECTION("Multiwavelet generation, degree = 3")
  {
    int const degree = 3;
    test_multiwavelet_gen<TestType>(degree, tol_factor);
  }
}

template<typename P>
void test_operator_two_scale(int const levels, int const degree)
{
  auto filename = transformations_base_dir /
                  ("operator_two_scale_" + std::to_string(degree + 1) + "_" +
                   std::to_string(levels) + ".dat");
  fk::matrix<P> const gold = read_matrix_from_txt_file<P>(filename);
  fk::matrix<P> const test = operator_two_scale<P>(degree, levels);

  auto constexpr tol_factor = get_tolerance<P>(100);

  rmse_comparison(gold, test, tol_factor);
}

TEMPLATE_TEST_CASE("operator_two_scale function working appropriately",
                   "[transformations]", test_precs)
{
  SECTION("operator_two_scale(2, 1)")
  {
    int const degree = 1;
    int const levels = 2;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(3, 1)")
  {
    int const degree = 1;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(3, 3)")
  {
    int const degree = 3;
    int const levels = 3;
    test_operator_two_scale<TestType>(levels, degree);
  }
  SECTION("operator_two_scale(5, 4)")
  {
    int const degree = 4;
    int const levels = 5;
    test_operator_two_scale<TestType>(levels, degree);
  }

  SECTION("operator_two_scale(6, 2)")
  {
    int const degree = 1;
    int const levels = 6;
    test_operator_two_scale<TestType>(levels, degree);
  }
}

template<typename P, resource resrc>
void test_fmwt_block_generation(int const level, int const degree)
{
  P constexpr tol = std::is_same_v<P, float> ? 1e-4 : 1e-13;

  prog_opts opts;
  opts.pde_choice = PDE_opts::diffusion_2; // not really relevant
  opts.start_levels = {level, };
  opts.degree = degree;
  auto const pde = make_PDE<P>(opts);

  basis::wavelet_transform<P, resrc> const forward_transform(*pde, verbosity_level::quiet);
  auto const &blocks = forward_transform.get_blocks();

  auto ctr = 0;
  for (auto const &block : blocks)
  {
    auto basis_base_dir        = gold_base_dir / "basis";
    std::string const gold_str = "transform_blocks_l" + std::to_string(level) +
                                 "_d" + std::to_string(degree + 1) + "_" +
                                 std::to_string(++ctr) + ".dat";
    fk::matrix<P> const gold =
        read_matrix_from_txt_file<P>(basis_base_dir / gold_str);

    if constexpr (resrc == resource::host)
    {
      rmse_comparison(gold, block, tol);
    }
    else
    {
      rmse_comparison(gold, block.clone_onto_host(), tol);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("wavelet constructor", "[basis]",

                       ((typename TestType, resource resrc), TestType, resrc),
                       mtest_precs)
{
  SECTION("level 2 degree 1")
  {
    auto const degree = 1;
    auto const levels = 2;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 2 degree 4")
  {
    auto const degree = 4;
    auto const levels = 2;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 1")
  {
    auto const degree = 1;
    auto const levels = 5;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 4")
  {
    auto const degree = 4;
    auto const levels = 5;
    test_fmwt_block_generation<TestType, resrc>(levels, degree);
  }
}

// tests transform across all supported levels
template<typename P, resource resrc>
void test_fmwt_application(int const level, int const degree)
{
  P constexpr tol = std::is_same_v<P, double> ? 1e-15 : 1e-5;

  prog_opts opts;
  opts.pde_choice = PDE_opts::diffusion_2; // irrelevant to the test
  opts.start_levels = {level,};
  opts.degree = degree;

  auto const pde = make_PDE<P>(opts);

  basis::wavelet_transform<P, resrc> const forward_transform(*pde, verbosity_level::quiet);

  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_real_distribution<P> dist(-2.0, 2.0);
  auto const gen = [&dist, &mersenne_engine]() {
    return dist(mersenne_engine);
  };

  for (auto l = 2; l <= forward_transform.max_level; ++l)
  {
    auto const dof = fm::ipow2(l) * (forward_transform.degree + 1);
    auto const to_transform = [&gen, dof]() {
      fk::matrix<P> matrix(dof, dof);
      std::generate(matrix.begin(), matrix.end(), gen);
      return matrix;
    }();

    auto const to_transform_v = [&gen, dof]() {
      fk::vector<P> vector(dof);
      std::generate(vector.begin(), vector.end(), gen);
      return vector;
    }();

    auto const fmwt = operator_two_scale<P>(forward_transform.degree, l);
    auto const fmwt_transpose = fk::matrix<P>(fmwt).transpose();

    auto const left_gold        = fmwt * to_transform;
    auto const right_gold       = to_transform * fmwt;
    auto const left_trans_gold  = fmwt_transpose * to_transform;
    auto const right_trans_gold = to_transform * fmwt_transpose;

    auto const left_gold_v        = fmwt * to_transform_v;
    auto const right_gold_v       = to_transform_v * fmwt;
    auto const left_trans_gold_v  = fmwt_transpose * to_transform_v;
    auto const right_trans_gold_v = to_transform_v * fmwt_transpose;

    if constexpr (resrc == resource::host)
    {
      auto const left_test = forward_transform.apply(
          to_transform, l, basis::side::left, basis::transpose::no_trans);
      auto const right_test = forward_transform.apply(
          to_transform, l, basis::side::right, basis::transpose::no_trans);
      auto const left_trans_test = forward_transform.apply(
          to_transform, l, basis::side::left, basis::transpose::trans);
      auto const right_trans_test = forward_transform.apply(
          to_transform, l, basis::side::right, basis::transpose::trans);

      auto const left_test_v = forward_transform.apply(
          to_transform_v, l, basis::side::left, basis::transpose::no_trans);
      auto const right_test_v = forward_transform.apply(
          to_transform_v, l, basis::side::right, basis::transpose::no_trans);
      auto const left_trans_test_v = forward_transform.apply(
          to_transform_v, l, basis::side::left, basis::transpose::trans);
      auto const right_trans_test_v = forward_transform.apply(
          to_transform_v, l, basis::side::right, basis::transpose::trans);

      rmse_comparison(left_test, left_gold, tol);
      rmse_comparison(right_test, right_gold, tol);
      rmse_comparison(left_trans_test, left_trans_gold, tol);
      rmse_comparison(right_trans_test, right_trans_gold, tol);

      rmse_comparison(left_test_v, left_gold_v, tol);
      rmse_comparison(right_test_v, right_gold_v, tol);
      rmse_comparison(left_trans_test_v, left_trans_gold_v, tol);
      rmse_comparison(right_trans_test_v, right_trans_gold_v, tol);
    }
    else
    {
      auto const transform_d = to_transform.clone_onto_device();
      auto const left_test   = forward_transform
                                 .apply(transform_d, l, basis::side::left,
                                        basis::transpose::no_trans)
                                 .clone_onto_host();
      auto const right_test = forward_transform
                                  .apply(transform_d, l, basis::side::right,
                                         basis::transpose::no_trans)
                                  .clone_onto_host();
      auto const left_trans_test =
          forward_transform
              .apply(transform_d, l, basis::side::left, basis::transpose::trans)
              .clone_onto_host();
      auto const right_trans_test =
          forward_transform
              .apply(transform_d, l, basis::side::right,
                     basis::transpose::trans)
              .clone_onto_host();

      auto const transform_dv = to_transform_v.clone_onto_device();
      auto const left_test_dv = forward_transform
                                    .apply(transform_dv, l, basis::side::left,
                                           basis::transpose::no_trans)
                                    .clone_onto_host();
      auto const right_test_dv = forward_transform
                                     .apply(transform_dv, l, basis::side::right,
                                            basis::transpose::no_trans)
                                     .clone_onto_host();
      auto const left_trans_test_dv =
          forward_transform
              .apply(transform_dv, l, basis::side::left,
                     basis::transpose::trans)
              .clone_onto_host();
      auto const right_trans_test_dv =
          forward_transform
              .apply(transform_dv, l, basis::side::right,
                     basis::transpose::trans)
              .clone_onto_host();

      rmse_comparison(left_test, left_gold, tol);
      rmse_comparison(right_test, right_gold, tol);
      rmse_comparison(left_trans_test, left_trans_gold, tol);
      rmse_comparison(right_trans_test, right_trans_gold, tol);

      rmse_comparison(left_test_dv, left_gold_v, tol);
      rmse_comparison(right_test_dv, right_gold_v, tol);
      rmse_comparison(left_trans_test_dv, left_trans_gold_v, tol);
      rmse_comparison(right_trans_test_dv, right_trans_gold_v, tol);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("wavelet transform", "[basis]",

                       ((typename TestType, resource resrc), TestType, resrc),
                       mtest_precs)
{
  SECTION("level 2 degree 1")
  {
    auto const degree = 1;
    auto const levels = 2;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }
  SECTION("level 2 degree 4")
  {
    auto const degree = 4;
    auto const levels = 2;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 2")
  {
    auto const degree = 2;
    auto const levels = 5;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }
  SECTION("level 5 degree 5")
  {
    auto const degree = 5;
    auto const levels = 5;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }

  SECTION("level 8 degree 3")
  {
    auto const degree = 3;
    auto const levels = 8;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }
  SECTION("level 8 degree 4")
  {
    auto const degree = 4;
    auto const levels = 8;

    test_fmwt_application<TestType, resrc>(levels, degree);
  }
}
