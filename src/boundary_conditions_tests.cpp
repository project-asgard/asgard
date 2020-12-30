#include "boundary_conditions.hpp"
#include "coefficients.hpp"
#include "elements.hpp"
#include "tests_general.hpp"

#include <regex>
template<typename P>
void test_boundary_condition_vector(PDE<P> &pde,
                                    std::string const &gold_filename,
                                    P const tol_factor)
{
  /* setup stuff */
  dimension<P> const &d = pde.get_dimensions()[0];
  int const level       = d.get_level();
  int const degree      = d.get_degree();

  auto const opts =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});
  elements::table const table(opts, pde);

  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);
  generate_all_coefficients<P>(pde, transformer);

  /* initialize bc vector at test_time */
  P const test_time = 0;

  int const start_element = 0;
  int const stop_element  = table.size() - 1;

  std::array<unscaled_bc_parts<P>, 2> unscaled_parts =
      boundary_conditions::make_unscaled_bc_parts(pde, table, transformer,
                                                  start_element, stop_element);

  fk::vector<P> bc_advanced = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, start_element, stop_element,
      test_time);

  fk::vector<P> const gold_bc_vector =
      fk::vector<P>(read_vector_from_txt_file(gold_filename));

  rmse_comparison(gold_bc_vector, bc_advanced, tol_factor);

  return;
}

template<typename P>
void test_compute_boundary_condition(PDE<P> &pde,
                                     std::string gold_filename_prefix,
                                     P const tol_factor)
{
  dimension<P> const &d = pde.get_dimensions()[0];

  // FIXME assume uniform degree
  int const level  = d.get_level();
  int const degree = d.get_degree();

  auto const opts =
      make_options({"-l", std::to_string(level), "-d", std::to_string(degree)});
  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);
  generate_all_coefficients<P>(pde, transformer);

  term_set<P> const &terms_vec_vec = pde.get_terms();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  elements::table const table(opts, pde);

  /* this timestep value must be consistent with the value used in the gold data
     generation scripts in matlab */
  P const time = 0;
  std::vector<std::vector<std::vector<fk::vector<P>>>> left_bc_parts;
  std::vector<std::vector<std::vector<fk::vector<P>>>> right_bc_parts;

  for (int term_num = 0; term_num < static_cast<int>(terms_vec_vec.size());
       ++term_num)
  {
    std::vector<term<P>> const &terms_vec = terms_vec_vec[term_num];
    for (int dim_num = 0; dim_num < static_cast<int>(dimensions.size());
         ++dim_num)
    {
      dimension<P> const &d = dimensions[dim_num];

      term<P> const &t = terms_vec[dim_num];

      std::vector<partial_term<P>> const &partial_terms = t.get_partial_terms();

      for (int p_num = 0; p_num < static_cast<int>(partial_terms.size());
           ++p_num)
      {
        std::string const gold_filename =
            gold_filename_prefix + "_bcL_" + std::to_string(d.get_degree()) +
            "d_" + std::to_string(d.get_level()) + "l_" +
            std::to_string(term_num) + "t_" + std::to_string(dim_num) + "dim_" +
            std::to_string(p_num) + "p.dat";
        partial_term<P> const &p_term = partial_terms[p_num];
        if (p_term.left_homo == homogeneity::inhomogeneous)
        {
          REQUIRE(static_cast<int>(p_term.left_bc_funcs.size()) > dim_num);

          fk::vector<P> const left_bc =
              boundary_conditions::compute_left_boundary_condition(
                  p_term.g_func, time, d, p_term.left_bc_funcs[dim_num]);

          /* compare to gold left bc */
          fk::vector<P> const gold_left_bc_vector =
              fk::vector<P>(read_vector_from_txt_file(gold_filename));
          rmse_comparison(gold_left_bc_vector, left_bc, tol_factor);
        }

        if (p_term.right_homo == homogeneity::inhomogeneous)
        {
          REQUIRE(static_cast<int>(p_term.right_bc_funcs.size()) > dim_num);

          fk::vector<P> const right_bc =
              boundary_conditions::compute_right_boundary_condition(
                  p_term.g_func, time, d, p_term.right_bc_funcs[dim_num]);
          /* compare to gold right bc */

          std::string const gold_right_filename =
              std::regex_replace(gold_filename, std::regex("bcL"), "bcR");
          fk::vector<P> const gold_right_bc_vector =
              fk::vector<P>(read_vector_from_txt_file(gold_right_filename));
          rmse_comparison(gold_right_bc_vector, right_bc, tol_factor);
        }
      }
    }
  }

  return;
}

TEMPLATE_TEST_CASE("problem separability", "[boundary_condition]", double,
                   float)
{
  /* intead of recalculating the boundary condition vectors at each timestep,
     calculate for the
     first and scale by multiplicative factors to at time + t */
  SECTION("time separability")
  {
    /* setup stuff */
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    auto const opts = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    elements::table const table(opts, *pde);

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);
    generate_all_coefficients<TestType>(*pde, transformer);

    /* initialize bc vector at test_time */
    TestType const test_time = 5;
    int const start_element  = 0;
    int const stop_element   = table.size() - 1;

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_1 =
        boundary_conditions::make_unscaled_bc_parts(
            *pde, table, transformer, start_element, stop_element, test_time);

    fk::vector<TestType> const bc_advanced_1 =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_1[0], unscaled_parts_1[1], *pde, start_element,
            stop_element, test_time);

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_0 =
        boundary_conditions::make_unscaled_bc_parts(
            *pde, table, transformer, start_element, stop_element);

    fk::vector<TestType> const bc_advanced_0 =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_0[0], unscaled_parts_0[1], *pde, start_element,
            stop_element, test_time);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-17 : 1e-8;
    rmse_comparison(bc_advanced_0, bc_advanced_1, tol_factor);
  }

  /* Intead of calculating the entire boundary condition vector, calculate a
   * portion */
  SECTION("element table split")
  {
    /* setup stuff */
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);
    auto const opts  = make_options(
        {"-l", std::to_string(level), "-d", std::to_string(degree)});

    elements::table const table(opts, *pde);

    basis::wavelet_transform<TestType, resource::host> const transformer(opts,
                                                                         *pde);
    generate_all_coefficients<TestType>(*pde, transformer);

    /* initialize bc vector at test_time */
    TestType const test_time = 0;

    int const start_element_0 = 0;
    int const stop_element_0  = table.size() - 1;

    std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts_0 =
        boundary_conditions::make_unscaled_bc_parts(*pde, table, transformer,
                                                    start_element_0,
                                                    stop_element_0, test_time);

    fk::vector<TestType> const bc_init =
        boundary_conditions::generate_scaled_bc(
            unscaled_parts_0[0], unscaled_parts_0[1], *pde, start_element_0,
            stop_element_0, test_time);

    /* create a vector for the first half of that vector */
    int index = 0;
    for (int table_element = 0; table_element < table.size(); ++table_element)
    {
      std::array<unscaled_bc_parts<TestType>, 2> const unscaled_parts =
          boundary_conditions::make_unscaled_bc_parts(
              *pde, table, transformer, table_element, table_element);

      fk::vector<TestType> const bc_advanced =
          boundary_conditions::generate_scaled_bc(
              unscaled_parts[0], unscaled_parts[1], *pde, table_element,
              table_element, test_time);

      fk::vector<TestType, mem_type::const_view> const bc_section(
          bc_init, index, index + bc_advanced.size() - 1);

      TestType const tol_factor =
          std::is_same<TestType, double>::value ? 1e-17 : 1e-4;
      rmse_comparison(bc_section, bc_advanced, tol_factor);

      index += bc_advanced.size();
    }
  }
}

TEMPLATE_TEST_CASE("compute_boundary_conditions", "[boundary_condition]",
                   double, float)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-15 : 1e-6;

  std::string const gold_filename_prefix =
      "../testing/generated-inputs/"
      "boundary_conditions/compute_diffusion1";
  SECTION("diffusion_1 level 2 degree 2")
  {
    int const level  = 2;
    int const degree = 2;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 4")
  {
    int const level  = 4;
    int const degree = 4;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }
}

TEMPLATE_TEST_CASE("boundary_conditions_vector", "[boundary_condition]", double,
                   float)
{
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-13 : 1e-4;

  SECTION("diffusion_1 level 2 degree 2")
  {
    int const level  = 2;
    int const degree = 2;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename = "../testing/generated-inputs/"
                                      "boundary_conditions/"
                                      "vector_diffusion1_l" +
                                      std::to_string(level) + "_d" +
                                      std::to_string(degree) + ".dat";

    test_boundary_condition_vector(*pde, gold_filename, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 4")
  {
    int const level  = 4;
    int const degree = 4;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename = "../testing/generated-inputs/"
                                      "boundary_conditions/"
                                      "vector_diffusion1_l" +
                                      std::to_string(level) + "_d" +
                                      std::to_string(degree) + ".dat";

    test_boundary_condition_vector(*pde, gold_filename, tol_factor);
  }
  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename = "../testing/generated-inputs/"
                                      "boundary_conditions/"
                                      "vector_diffusion1_l" +
                                      std::to_string(level) + "_d" +
                                      std::to_string(degree) + ".dat";

    test_boundary_condition_vector(*pde, gold_filename, tol_factor);
  }
}
