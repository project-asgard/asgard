#include "pde.hpp"

#include "matlab_utilities.hpp"
#include "tests_general.hpp"
#include <vector>

static auto const pde_eps_multiplier = 1e2;

static auto const pde_base_dir = gold_base_dir / "pde";

template<typename P>
void test_initial_condition(PDE<P> const &pde, std::filesystem::path base_dir,
                            fk::vector<P> const &x)
{
  auto const filename = base_dir.filename().string();
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const gold =
        fk::vector<P>(read_vector_from_txt_file(base_dir.replace_filename(
            filename + "initial_dim" + std::to_string(i) + ".dat")));
    auto const fx = pde.get_dimensions()[i].initial_condition(x, 0);

    auto constexpr tol_factor = get_tolerance<P>(10);

    rmse_comparison(fx, gold, tol_factor);
  }
}

template<typename P>
void test_exact_solution(PDE<P> const &pde, std::filesystem::path base_dir,
                         fk::vector<P> const &x, P const time)
{
  if (!pde.has_analytic_soln)
  {
    return;
  }

  auto constexpr tol_factor = get_tolerance<P>(10);
  auto const filename       = base_dir.filename().string();
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const gold =
        fk::vector<P>(read_vector_from_txt_file(base_dir.replace_filename(
            filename + "exact_dim" + std::to_string(i) + ".dat")));
    auto const fx = pde.exact_vector_funcs[i](x, time);
    rmse_comparison(fx, gold, tol_factor);
  }

  P const gold = read_scalar_from_txt_file(
      base_dir.replace_filename(filename + "exact_time.dat"));
  P const fx = pde.exact_time(time);
  relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
}

template<typename P>
void test_source_vectors(PDE<P> const &pde, std::filesystem::path base_dir,
                         fk::vector<P> const &x, P const time)
{
  auto constexpr tol_factor = get_tolerance<P>(10);
  auto const filename       = base_dir.filename().string();

  for (auto i = 0; i < pde.num_sources; ++i)
  {
    auto const source_string = filename + "source" + std::to_string(i) + "_";
    for (auto j = 0; j < pde.num_dims; ++j)
    {
      auto const full_path = base_dir.replace_filename(
          source_string + "dim" + std::to_string(j) + ".dat");
      auto const gold = fk::vector<P>(read_vector_from_txt_file(full_path));
      auto const fx   = pde.sources[i].source_funcs[j](x, time);
      rmse_comparison(fx, gold, tol_factor);
    }
    P const gold = read_scalar_from_txt_file(
        base_dir.replace_filename(source_string + "time.dat"));
    auto const fx = pde.sources[i].time_func(time);
    relaxed_fp_comparison(fx, gold, pde_eps_multiplier);
  }
}

TEMPLATE_TEST_CASE("testing diffusion 2 implementations", "[pde]", double,
                   float)
{
  auto const level  = 3;
  auto const degree = 2;
  auto const pde    = make_PDE<TestType>(PDE_opts::diffusion_2, level, degree);
  auto const base_dir          = pde_base_dir / "diffusion_2_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("diffusion 2 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("diffusion 2 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("diffusion 2 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing diffusion 1 implementations", "[pde]", double,
                   float)
{
  auto const level  = 3;
  auto const degree = 2;
  auto const pde    = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
  auto const base_dir          = pde_base_dir / "diffusion_1_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("diffusion 1 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("diffusion 1 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("diffusion 1 source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("diffusion 1 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 1 implementations", "[pde]", double,
                   float)
{
  auto const pde               = make_PDE<TestType>(PDE_opts::continuity_1);
  auto const base_dir          = pde_base_dir / "continuity_1_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("continuity 1 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("continuity 1 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("continuity 1 source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("continuity 1 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}
TEMPLATE_TEST_CASE("testing contuinity 2 implementations, level 5, degree 4",
                   "[pde]", double, float)
{
  auto const level  = 5;
  auto const degree = 4;
  auto const pde    = make_PDE<TestType>(PDE_opts::continuity_2, level, degree);
  auto const base_dir          = pde_base_dir / "continuity2_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("continuity 2 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("continuity 2 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("continuity 2 source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("continuity 2 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 3 implementations", "[pde]", double,
                   float)
{
  auto const level  = 5;
  auto const degree = 4;
  auto const pde    = make_PDE<TestType>(PDE_opts::continuity_3, level, degree);
  auto const base_dir          = pde_base_dir / "continuity_3_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("continuity 3 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("continuity 3 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("continuity 3 source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("continuity 3 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing contuinity 6 implementations", "[pde]", double,
                   float)
{
  auto const level    = 3;
  auto const pde      = make_PDE<TestType>(PDE_opts::continuity_6, level);
  auto const base_dir = pde_base_dir / "continuity_6_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("continuity 6 initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("continuity 6 exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("continuity 6 source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("continuity 6 dt")
  {
    auto filename = base_dir.filename().string();
    TestType const gold =
        read_scalar_from_txt_file(pde_base_dir / (filename + "dt.dat"));
    TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    REQUIRE(dt == gold);
  }
}

TEMPLATE_TEST_CASE("testing fokkerplanck2_complete_case4 implementations",
                   "[pde]", double, float)
{
  int const level  = 5;
  int const degree = 4;

  auto const pde = make_PDE<TestType>(PDE_opts::fokkerplanck_2d_complete_case4,
                                      level, degree);
  auto const base_dir          = pde_base_dir / "fokkerplanck2_complete_";
  fk::vector<TestType> const x = {0.1, 0.2, 0.3, 0.4, 0.5};
  TestType const time          = 5;

  SECTION("fp2 complete initial condition functions")
  {
    test_initial_condition<TestType>(*pde, base_dir, x);
  }

  SECTION("fp2 complete exact solution functions")
  {
    test_exact_solution<TestType>(*pde, base_dir, x, time);
  }

  SECTION("fp2 complete source functions")
  {
    test_source_vectors(*pde, base_dir, x, time);
  }

  SECTION("fp2 complete dt")
  {
    // TestType const gold = read_scalar_from_txt_file(base_dir + "dt.dat");
    // TestType const dt = pde->get_dt() / parser::DEFAULT_CFL;
    // REQUIRE(dt == gold); // not testing this for now
    // different domain mins between matlab/C++ will produce different dts
  }

  SECTION("fp2 complete pterm funcs")
  {
    auto filename   = base_dir.filename().string();
    auto const gold = fk::matrix<TestType>(
        read_matrix_from_txt_file(pde_base_dir / (filename + "gfuncs.dat")));
    auto const gold_dvs = fk::matrix<TestType>(
        read_matrix_from_txt_file(pde_base_dir / (filename + "dvfuncs.dat")));

    int row = 0;
    for (auto i = 0; i < pde->num_dims; ++i)
    {
      for (auto j = 0; j < pde->num_terms; ++j)
      {
        auto const &term_1D       = pde->get_terms()[j][i];
        auto const &partial_terms = term_1D.get_partial_terms();
        for (auto k = 0; k < static_cast<int>(partial_terms.size()); ++k)
        {
          fk::vector<TestType> transformed(x);
          g_func_type<TestType> g_func = partial_terms[k].g_func;
          std::transform(x.begin(), x.end(), transformed.begin(),
                         [g_func, time](TestType const x_elem) -> TestType {
                           return g_func(x_elem, time);
                         });
          fk::vector<TestType> gold_pterm(
              gold.extract_submatrix(row, 0, 1, x.size()));
          auto constexpr tol_factor = get_tolerance<TestType>(100);
          rmse_comparison(transformed, gold_pterm, tol_factor);

          fk::vector<TestType> dv(x);
          g_func_type<TestType> dv_func = partial_terms[k].dv_func;
          std::transform(x.begin(), x.end(), dv.begin(),
                         [dv_func, time](TestType const x_elem) -> TestType {
                           return dv_func(x_elem, time);
                         });
          fk::vector<TestType> gold_dvfunc(
              gold_dvs.extract_submatrix(row, 0, 1, x.size()));
          rmse_comparison(dv, gold_dvfunc, tol_factor);

          row++;
        }
      }
    }
  }
}
