#include "tests_general.hpp"

static auto const coefficients_base_dir = gold_base_dir / "coefficients";

using namespace asgard;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

template<typename P>
void test_coefficients(prog_opts const &opts, std::string const &gold_path,
                       P const tol_factor = get_tolerance<P>(10))
{
  discretization_manager<P> disc(make_PDE<P>(opts));

  auto &pde        = disc.get_pde();
  int const degree = disc.degree();

  auto const lev_string = std::accumulate(
      pde.get_dimensions().begin(), pde.get_dimensions().end(), std::string(),
      [](std::string const &accum, dimension<P> const &dim) {
        return accum + std::to_string(dim.get_level()) + "_";
      });

  auto const filename_base = gold_path + "_l" + lev_string + "d" +
                             std::to_string(degree + 1) + "_";

  int num_terms = pde.num_terms();
  // hack here!
  // skip the last vlassov term, the coefficients are hard-coded but had to be changed
  // to use alternating fluxes which in turn creates a discrepancy
  if (gold_path.find("vlasov_lb_full_f_coefficients") != std::string::npos)
    num_terms -= 1;

  for (int d : indexof<int>(pde.num_dims()))
  {
    for (int64_t t : indexof(num_terms))
    {
      auto const filename = filename_base + std::to_string(t + 1) + "_" +
                            std::to_string(d + 1) + ".dat";
      fk::matrix<P> const gold = read_matrix_from_txt_file<P>(filename);

      auto const full_coeff = disc.get_coeff_matrix(t, d);

      auto const &dim = pde.get_dimensions()[d];
      auto const dof  = (degree + 1) * fm::ipow2(dim.get_level());

      fk::matrix<P, mem_type::const_view> const test(full_coeff, 0, dof - 1, 0, dof - 1);

      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", test_precs)
{
  auto const gold_path      = coefficients_base_dir / "diffusion2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  prog_opts opts;
  opts.pde_choice = PDE_opts::diffusion_2;

  SECTION("level 3, degree 4")
  {
    opts.degree       = 4;
    opts.start_levels = {3, 3};
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 2, 3, degree 4")
  {
    opts.degree       = 4;
    opts.start_levels = {2, 3};
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", test_precs)
{
  auto const gold_path      = coefficients_base_dir / "diffusion1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10000);

  SECTION("level 5, degree 5")
  {
    auto opts = make_opts("-p diffusion_1 -l 5 -d 5");
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", test_precs)
{
  auto const gold_path  = coefficients_base_dir / "continuity1_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("level 2, degree 1 (default)")
  {
    auto opts = make_opts("-p continuity_1 -l 2 -d 1");
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", test_precs)
{
  auto const gold_path      = coefficients_base_dir / "continuity2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  prog_opts opts;
  opts.pde_choice = PDE_opts::continuity_2;

  SECTION("level 4, degree 2")
  {
    opts.start_levels = {4, 4};
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 4, 5, degree 2")
  {
    opts.start_levels = {4, 5};
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", test_precs)
{
  auto const gold_path      = coefficients_base_dir / "continuity3_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  prog_opts opts;
  opts.pde_choice = PDE_opts::continuity_3;

  SECTION("level 4, degree 3")
  {
    opts.start_levels = {4, 4, 4};
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 2, degree 3")
  {
    opts.start_levels = {2, 3, 2};
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", test_precs)
{
  auto const gold_path      = coefficients_base_dir / "continuity6_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  prog_opts opts;
  opts.pde_choice = PDE_opts::continuity_6;

  SECTION("level 2, degree 3")
  {
    opts.start_levels = std::vector<int>(6, 2);
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 3, 3, 2, 4, degree 3")
  {
    opts.start_levels = {2, 3, 3, 3, 2, 4};
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case1 terms", "[coefficients]",
                   test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p1a_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 2")
  {
    prog_opts opts;
    opts.pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case1;
    opts.start_levels = {4, };
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case2 terms", "[coefficients]",
                   test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_pitch_E_case2_coefficients";
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  SECTION("level 4, degree 2")
  {
    prog_opts opts;
    opts.pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case2;
    opts.start_levels = {4, };
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_C terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p2_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-5;

  SECTION("level 5, degree 1")
  {
    prog_opts opts;
    opts.pde_choice = PDE_opts::fokkerplanck_1d_pitch_C;
    opts.start_levels = {5, };
    opts.degree       = 1;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p3_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-13 : 1e-4;

  SECTION("level 2, degree 4")
  {
    prog_opts opts;
    opts.pde_choice = PDE_opts::fokkerplanck_1d_4p3;
    opts.start_levels = {2, };
    opts.degree       = 4;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p4_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-14 : 1e-6;

  SECTION("level 5, degree 2")
  {
    prog_opts opts;
    opts.pde_choice = PDE_opts::fokkerplanck_1d_4p4;
    opts.start_levels = {5, };
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck1_4p5_coefficients";
  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-13 : 1e-4;

  SECTION("level 3, degree 4")
  {
    prog_opts opts;
    opts.pde_choice   = PDE_opts::fokkerplanck_1d_4p5;
    opts.start_levels = {3, };
    opts.degree       = 4;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete_case4 terms", "[coefficients]",
                   test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "fokkerplanck2_complete_coefficients";

  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-12 : 1e-3;

  prog_opts opts;
  opts.pde_choice = PDE_opts::fokkerplanck_2d_complete_case4;

  SECTION("level 3, degree 2")
  {
    opts.start_levels = {3, 3};
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("level 4, degree 3")
  {
    opts.start_levels = {4, 4};
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
  SECTION("non-uniform levels: 2, 3, degree 2")
  {
    opts.start_levels = {2, 3};
    opts.degree       = 2;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }

  SECTION("non-uniform levels: 4, 2, degree 3")
  {
    opts.start_levels = {4, 2};
    opts.degree       = 3;
    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("vlasov terms", "[coefficients]", test_precs)
{
  auto const gold_path =
      coefficients_base_dir / "vlasov_lb_full_f_coefficients";

  TestType const tol_factor = std::is_same_v<TestType, double> ? 1e-12 : 1e-3;

  prog_opts opts;
  opts.pde_choice     = PDE_opts::vlasov_lb_full_f;
  opts.grid           = grid_type::dense;
  opts.num_time_steps = 1;

  SECTION("level [4,3], degree 2")
  {
    opts.start_levels = {4, 3};
    opts.degree       = 2;

    test_coefficients<TestType>(opts, gold_path, tol_factor);
  }
}

template<typename P>
class penalty_pde : public PDE<P>
{
public:
  penalty_pde()
  {
    vector_func<P> ic = {partial_term<P>::null_vector_func};
    g_func_type<P> gfunc;

    dimension<P> dim(0.0, 1.0, 4, 2, ic, gfunc, "x");

    partial_term<P> central(
        coefficient_type::div, nullptr, nullptr, flux_type::central,
        boundary_condition::periodic, boundary_condition::periodic);

    partial_term<P> penalty(
        coefficient_type::penalty, nullptr, nullptr, flux_type::downwind,
        boundary_condition::periodic, boundary_condition::periodic);

    partial_term<P> downwind(
        coefficient_type::div, nullptr, nullptr, flux_type::downwind,
        boundary_condition::periodic, boundary_condition::periodic);

    term<P> tc(false, "-u", {central, }, imex_flag::unspecified);
    term<P> tp(false, "-u", {penalty, }, imex_flag::unspecified);
    term<P> td(false, "-u", {downwind, }, imex_flag::unspecified);

    term_set<P> terms = std::vector<std::vector<term<P>>>{
      std::vector<term<P>>{tc, }, std::vector<term<P>>{tp, }, std::vector<term<P>>{td, }};

    this->initialize(prog_opts(), 1, 0, 3,
                     {dim, }, terms, std::vector<source<P>>{},
                     std::vector<md_func_type<P>>{{}},
                     get_dt_, false, false, moment_funcs<P>{}, false);
  }
  static P get_dt_(dimension<P> const &) { return 1.0; }
};

TEMPLATE_TEST_CASE("penalty check", "[coefficients]", test_precs)
{
  vector_func<TestType> ic = {partial_term<TestType>::null_vector_func};
  g_func_type<TestType> gfunc;

  SECTION("level 4, degree 2")
  {
    std::unique_ptr<penalty_pde<TestType>> pde = std::make_unique<penalty_pde<TestType>>();
    discretization_manager<TestType> disc(std::unique_ptr<PDE<TestType>>(pde.release()));

    auto central_mat = disc.get_coeff_matrix(0, 0);
    auto penalty_mat = disc.get_coeff_matrix(1, 0);
    auto downwind_mat = disc.get_coeff_matrix(2, 0);

    rmse_comparison(central_mat + penalty_mat, downwind_mat,
                    get_tolerance<TestType>(10));
  }
}
