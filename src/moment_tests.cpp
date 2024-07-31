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

TEMPLATE_TEST_CASE("Multiwavelet", "[transformations]", test_precs)
{
  std::string const pde_choice = "diffusion_2";
  fk::vector<int> const levels{5, 5};

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, 3);
  parser_mod::set(parse, parser_mod::cfl, 0.01);
  parser_mod::set(parse, parser_mod::use_full_grid, false);
  parser_mod::set(parse, parser_mod::num_time_steps, 5);
  parser_mod::set(parse, parser_mod::use_implicit_stepping, true);
  parser_mod::set(parse, parser_mod::do_adapt, true);
  parser_mod::set(parse, parser_mod::adapt_threshold, 0.5e-1);

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

TEMPLATE_TEST_CASE("CreateMomentReducedMatrix", "[moments]", test_precs)
{
  std::string const pde_choice = "vlasov";
  fk::vector<int> const levels{4, 3};
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  parser parse(pde_choice, levels);
  parser_mod::set(parse, parser_mod::degree, 2);
  parser_mod::set(parse, parser_mod::cfl, 0.01);
  parser_mod::set(parse, parser_mod::use_full_grid, true);
  parser_mod::set(parse, parser_mod::num_time_steps, 1);
  parser_mod::set(parse, parser_mod::use_implicit_stepping, false);
  parser_mod::set(parse, parser_mod::do_adapt, false);
  parser_mod::set(parse, parser_mod::adapt_threshold, 0.5e-1);

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
    moments[i].createMomentVector(*pde, opts, check);
    moments[i].createMomentReducedMatrix(*pde, check);

    auto const gold_filename =
        moment_base_dir /
        ("moment_matrix_vlasov_d3_l4_3_m" + std::to_string(i + 1) + ".dat");
    auto const gold_moment_matrix =
        read_matrix_from_txt_file<TestType>(gold_filename);

#ifdef ASGARD_USE_CUDA
    rmse_comparison(
        gold_moment_matrix,
        moments[i].get_moment_matrix_dev().clone_onto_host().to_dense(),
        tol_factor);
#else
    rmse_comparison(gold_moment_matrix,
                    moments[i].get_moment_matrix_dev().to_dense(), tol_factor);
#endif
  }
}
