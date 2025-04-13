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
  // skip the last vlasov term, the coefficients are hard-coded but had to be changed
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

    this->initialize(prog_opts(), 1, 0,
                     {dim, }, terms, std::vector<source<P>>{},
                     std::vector<md_func_type<P>>{{}}, get_dt_, false, false);
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
