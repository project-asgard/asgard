#include "adapt.hpp"
#include "basis.hpp"
#include "coefficients.hpp"
#include "moment.hpp"
#include "tests_general.hpp"

static auto const moment_base_dir = gold_base_dir / "moment";

using namespace asgard;

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", double, float)
{
  std::string const pde_choice = "diffusion_2";
  fk::vector<int> const levels{5, 5};
  auto const degree               = 4;
  auto const cfl                  = 0.01;
  auto const full_grid            = false;
  static auto constexpr num_steps = 5;
  auto const use_implicit         = true;
  auto const do_adapt_levels      = true;
  auto const adapt_threshold      = 0.5e-1;

  parser const parse(pde_choice, levels, degree, cfl, full_grid,
                     parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  auto pde = make_PDE<TestType>(parse);
  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  std::vector<vector_func<TestType>> md_func;
  SECTION("Constructor") { moment<TestType> mymoment({md_func}); }
}

TEMPLATE_TEST_CASE("CreateMomentReducedMatrix", "[moments]", double, float)
{
  std::string const pde_choice = "vlasov";
  fk::vector<int> const levels{4, 3};
  auto const degree               = 3;
  auto const cfl                  = 0.01;
  auto const full_grid            = true;
  static auto constexpr num_steps = 1;
  auto const use_implicit         = false;
  auto const do_adapt_levels      = false;
  auto const adapt_threshold      = 0.5e-1;

  auto constexpr tol_factor = get_tolerance<TestType>(100);

  parser const parse(pde_choice, levels, degree, cfl, full_grid,
                     parser::DEFAULT_MAX_LEVEL, num_steps, use_implicit,
                     do_adapt_levels, adapt_threshold);

  auto pde = make_PDE<TestType>(parse);
  options const opts(parse);
  elements::table const check(opts, *pde);

  adapt::distributed_grid adaptive_grid(*pde, opts);
  basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                       *pde);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);

  auto moments = pde->moments;
  REQUIRE(moments.size() > 0);

  for (size_t i = 0; i < moments.size(); ++i)
  {
    moments[i].createFlist(*pde, opts);
    moments[i].createMomentVector(*pde, parse, check);
    moments[i].createMomentReducedMatrix(*pde, check);

    auto const gold_filename =
        moment_base_dir /
        ("moment_matrix_vlasov_d3_l4_3_m" + std::to_string(i + 1) + ".dat");
    auto const gold_moment_matrix =
        fk::matrix<TestType>(read_matrix_from_txt_file(gold_filename));

    rmse_comparison(gold_moment_matrix, moments[i].get_moment_matrix(),
                    tol_factor);
  }
}
