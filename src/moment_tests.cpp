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

  auto pde = make_PDE<TestType>("-p diffusion_2 -l 5 -d 3 -n 5 -sm impl -a 0.5e-1");
  //options const opts(parse);
  elements::table const check(*pde);

  adapt::distributed_grid adaptive_grid(*pde);
  basis::wavelet_transform<TestType, resource::host> const transformer(*pde);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer);

  std::vector<vector_func<TestType>> md_func;
  SECTION("Constructor") { moment<TestType> mymoment({md_func}); }
  // this is compile/crash test, if we got this fast we're good
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("CreateMomentReducedMatrix", "[moments]", test_precs)
{
  std::string const pde_choice = "vlasov";
  fk::vector<int> const levels{4, 3};
  auto constexpr tol_factor = get_tolerance<TestType>(100);

  prog_opts opts;
  opts.pde_choice     = PDE_opts::vlasov_lb_full_f;
  opts.start_levels   = {4, 3};
  opts.degree         = 2;
  opts.grid           = grid_type::dense;
  opts.num_time_steps = 1;

  auto pde = make_PDE<TestType>(opts);
  elements::table const check(*pde);

  adapt::distributed_grid adaptive_grid(*pde);
  basis::wavelet_transform<TestType, resource::host> const transformer(*pde);

  // -- set coeffs
  generate_all_coefficients(*pde, transformer);

  // -- generate initial condition vector.
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer);

  auto moments = pde->moments;
  REQUIRE(moments.size() > 0);

  for (size_t i = 0; i < moments.size(); ++i)
  {
    moments[i].createFlist(*pde);
    moments[i].createMomentVector(*pde, check);
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
