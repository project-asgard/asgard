#include "tests_general.hpp"

static auto const boundary_conditions_base_dir =
    gold_base_dir / "boundary_conditions";

using namespace asgard;
using namespace asgard::boundary_conditions;

int main(int argc, char *argv[])
{
  initialize_distribution();

  int result = Catch::Session().run(argc, argv);

  finalize_distribution();

  return result;
}

template<typename P>
void test_boundary_condition_vector(std::string const &opts_str,
                                    std::filesystem::path const &gold_filename,
                                    P const tol_factor)
{
  discretization_manager<P> disc(make_PDE<P>(opts_str));
  auto const &pde = disc.get_pde();
  elements::table const table(pde);

  basis::wavelet_transform<P, resource::host> const &transformer = disc.get_transformer();

  /* initialize bc vector at test_time */
  P const test_time = 0;

  int const start_element = 0;
  int const stop_element  = table.size() - 1;

  std::array<unscaled_bc_parts<P>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(
          pde, table, transformer, disc.get_hiermanip(), disc.get_cmatrices(),
          disc.get_conn(), start_element, stop_element);

  fk::vector<P> bc_advanced = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, start_element, stop_element,
      test_time);

  fk::vector<P> const gold_bc_vector =
      read_vector_from_txt_file<P>(gold_filename);

  rmse_comparison(gold_bc_vector, bc_advanced, tol_factor);

  return;
}

template<typename P>
void test_compute_boundary_condition(PDE<P> &pde,
                                     std::string gold_filename_prefix,
                                     P const tol_factor)
{
  term_set<P> const &terms_vec_vec = pde.get_terms();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  int const num_dims = pde.num_dims();

  elements::table const table(pde);

  /* this time-step value must be consistent with the value used in the gold data
     generation scripts in matlab */
  P const time = 0;
  std::vector<std::vector<std::vector<fk::vector<P>>>> left_bc_parts;
  std::vector<std::vector<std::vector<fk::vector<P>>>> right_bc_parts;

  for (int t : indexof<int>(terms_vec_vec.size()))
  {
    std::vector<term<P>> const &term_md = terms_vec_vec[t];
    for (int d : indexof<int>(num_dims))
    {
      dimension<P> const &dim = dimensions[d];

      std::vector<partial_term<P>> const &partial_terms = term_md[d].get_partial_terms();

      int const pdof = dim.get_degree() + 1;

      auto gen_filename = [&](std::string const &bc, int pt)
          -> std::string {
        return gold_filename_prefix + '_' + bc + '_' + std::to_string(pdof) +
               "d_" + std::to_string(dim.get_level()) + "l_" +
               std::to_string(t) + "t_" + std::to_string(d) + "dim_" +
               std::to_string(pt) + "p.dat";
      };

      for (int pt : indexof<int>(partial_terms.size()))
      {
        std::string const gold_filename = gen_filename("bcL", pt);

        partial_term<P> const &p_term = partial_terms[pt];
        if (p_term.left_homo() == homogeneity::inhomogeneous)
        {
          REQUIRE(static_cast<int>(p_term.left_bc_funcs().size()) > d);

          fk::vector<P> const left_bc =
              boundary_conditions::compute_left_boundary_condition(
                  p_term.g_func(), p_term.dv_func(), time, dim,
                  p_term.left_bc_funcs()[d]);

          /* compare to gold left bc */
          fk::vector<P> const gold_left_bc_vector =
              read_vector_from_txt_file<P>(gold_filename);
          rmse_comparison(gold_left_bc_vector, left_bc, tol_factor);
        }

        if (p_term.right_homo() == homogeneity::inhomogeneous)
        {
          REQUIRE(static_cast<int>(p_term.right_bc_funcs().size()) > d);

          fk::vector<P> const right_bc =
              boundary_conditions::compute_right_boundary_condition(
                  p_term.g_func(), p_term.dv_func(), time, dim,
                  p_term.right_bc_funcs()[d]);
          /* compare to gold right bc */

          std::string const gold_right_filename = gen_filename("bcR", pt);

          fk::vector<P> const gold_right_bc_vector =
              read_vector_from_txt_file<P>(gold_right_filename);
          rmse_comparison(gold_right_bc_vector, right_bc, tol_factor);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("problem separability", "[boundary_condition]", test_precs)
{
  /* instead of recalculating the boundary condition vectors at each timestep,
     calculate for the
     first and scale by multiplicative factors to at time + t */
  SECTION("time separability")
  {
    discretization_manager<TestType> disc(make_PDE<TestType>("-p diffusion_1 -l 5 -d 4"));
    auto const &pde = disc.get_pde();

    elements::table const table(pde);

    basis::wavelet_transform<TestType, resource::host> const &transformer = disc.get_transformer();

    // initialize bc vector at test_time
    TestType const test_time = 5;
    int const start_element  = 0;
    int const stop_element   = table.size() - 1;

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_1 =
        boundary_conditions::make_unscaled_bc_parts(
            pde, table, transformer, disc.get_hiermanip(), disc.get_cmatrices(),
            disc.get_conn(), start_element, stop_element, test_time);

    fk::vector<TestType> const bc_advanced_1 =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_1[0], unscaled_parts_1[1], pde, start_element,
            stop_element, test_time);

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_0 =
        boundary_conditions::make_unscaled_bc_parts(
            pde, table, transformer, disc.get_hiermanip(), disc.get_cmatrices(),
            disc.get_conn(), start_element, stop_element);

    fk::vector<TestType> const bc_advanced_0 =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_0[0], unscaled_parts_0[1], pde, start_element,
            stop_element, test_time);

    auto constexpr tol_factor = get_tolerance<TestType>(10);
    rmse_comparison(bc_advanced_0, bc_advanced_1, tol_factor);
  }

  /* Instead of calculating the entire boundary condition vector, calculate a
   * portion */
  SECTION("element table split")
  {
    /* setup stuff */
    discretization_manager<TestType> disc(make_PDE<TestType>("-p diffusion_1 -l 5 -d 5"));
    auto const &pde = disc.get_pde();

    elements::table const table(pde);

    basis::wavelet_transform<TestType, resource::host> const &transformer = disc.get_transformer();

    /* initialize bc vector at test_time */
    TestType const test_time = 0;

    int const start_element_0 = 0;
    int const stop_element_0  = table.size() - 1;

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_0 =
        boundary_conditions::make_unscaled_bc_parts(
            pde, table, transformer, disc.get_hiermanip(), disc.get_cmatrices(),
            disc.get_conn(), start_element_0, stop_element_0, test_time);

    fk::vector<TestType> const bc_init =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_0[0], unscaled_parts_0[1], pde, start_element_0,
            stop_element_0, test_time);

    /* create a vector for the first half of that vector */
    int index = 0;
    for (int table_element = 0; table_element < table.size(); ++table_element)
    {
      std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts =
          boundary_conditions::make_unscaled_bc_parts(
              pde, table, transformer, disc.get_hiermanip(), disc.get_cmatrices(),
              disc.get_conn(), table_element, table_element);

      fk::vector<TestType> const bc_advanced =
          boundary_conditions::generate_scaled_bc(
              unscaled_parts[0], unscaled_parts[1], pde, table_element,
              table_element, test_time);

      fk::vector<TestType, mem_type::const_view> const bc_section(
          bc_init, index, index + bc_advanced.size() - 1);

      auto constexpr tol_factor = get_tolerance<TestType>(1e4);

      rmse_comparison(bc_section, bc_advanced, tol_factor);

      index += bc_advanced.size();
    }
  }
}

TEMPLATE_TEST_CASE("compute_boundary_conditions", "[boundary_condition]",
                   test_precs)
{
  auto constexpr tol_factor = get_tolerance<TestType>(10);

  auto const gold_filename_prefix =
      boundary_conditions_base_dir / "compute_diffusion1";

  SECTION("diffusion_1 level 2 degree 1")
  {
    auto const pde = make_PDE<TestType>("-p diffusion_1 -l 2 -d 1");

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 3")
  {
    auto const pde = make_PDE<TestType>("-p diffusion_1 -l 4 -d 3");

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 5 degree 4")
  {
    auto const pde = make_PDE<TestType>("-p diffusion_1 -l 5 -d 4");

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_2 level 3 degree 2")
  {
    auto const pde = make_PDE<TestType>("-p diffusion_2 -l 3 -d 2");

    auto const gold_prefix =
        boundary_conditions_base_dir / "compute_diffusion2";

    test_compute_boundary_condition(*pde, gold_prefix, tol_factor);
  }
}

TEMPLATE_TEST_CASE("boundary_conditions_vector", "[boundary_condition]",
                   test_precs)
{
  auto constexpr tol_factor = get_tolerance<TestType>(1000);

  SECTION("diffusion_1 level 2 degree 1")
  {
    int const level  = 2;
    int const degree = 1;

    auto const gold_filename = boundary_conditions_base_dir /
                               ("vector_diffusion1_l" + std::to_string(level) +
                                "_d" + std::to_string(degree + 1) + ".dat");

    test_boundary_condition_vector("-p diffusion_1 -l 2 -d 1", gold_filename, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 3")
  {
    int const level  = 4;
    int const degree = 3;

    auto const gold_filename = boundary_conditions_base_dir /
                               ("vector_diffusion1_l" + std::to_string(level) +
                                "_d" + std::to_string(degree + 1) + ".dat");

    test_boundary_condition_vector("-p diffusion_1 -l 4 -d 3", gold_filename, tol_factor);
  }
  SECTION("diffusion_1 level 5 degree 4")
  {
    int const level  = 5;
    int const degree = 4;

    auto const gold_filename = boundary_conditions_base_dir /
                               ("vector_diffusion1_l" + std::to_string(level) +
                                "_d" + std::to_string(degree + 1) + ".dat");

    test_boundary_condition_vector("-p diffusion_1 -l 5 -d 4", gold_filename, tol_factor);
  }

  SECTION("diffusion_2 level 3 degree 2")
  {
    int const level  = 3;
    int const degree = 2;

    auto const gold_filename = boundary_conditions_base_dir /
                               ("vector_diffusion2_l" + std::to_string(level) +
                                "_d" + std::to_string(degree + 1) + ".dat");

    test_boundary_condition_vector("-p diffusion_2 -l 3 -d 2", gold_filename, tol_factor);
  }
}
