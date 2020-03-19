#include "boundary_conditions.hpp"
#include "coefficients.hpp"
#include "element_table.hpp"
#include "tests_general.hpp"

template<typename P>
void test_boundary_condition_vector(PDE<P> &pde,
                                    std::string const &gold_filename_prefix,
                                    double const eps_multiplier)
{
  /* setup stuff */
  dimension<P> const &d = pde.get_dimensions()[0];
  int const level       = d.get_level();
  int const degree      = d.get_degree();

  element_table table(make_options({"-l", std::to_string(level)}),
                      pde.num_dims);

  generate_all_coefficients<P>(pde);

  /* initialize bc vector at test_time */
  P const test_time = 0;

  std::vector<std::vector<std::vector<fk::vector<P>>>> left_bc_parts;
  std::vector<std::vector<std::vector<fk::vector<P>>>> right_bc_parts;
  int const start_element = 0;
  int const stop_element  = table.size() - 1;

  boundary_conditions::init_bc_parts(pde, table, start_element, stop_element,
                                     left_bc_parts, right_bc_parts);

  fk::vector<P> bc_advanced =
      boundary_conditions::generate_bc(left_bc_parts, right_bc_parts, pde,
                                       start_element, stop_element, test_time);

  std::string const gold_filename =
      gold_filename_prefix + "boundary_condition_vector" + "_l" +
      std::to_string(level) + "_d" + std::to_string(degree) + ".dat";

  fk::vector<P> const gold_bc_vector =
      fk::vector<P>(read_vector_from_txt_file(gold_filename));

  relaxed_comparison(gold_bc_vector, bc_advanced, eps_multiplier);

  return;
}

template<typename P>
void test_compute_boundary_condition(PDE<P> &pde,
                                     std::string gold_filename_prefix,
                                     double const eps_multiplier)
{
  generate_all_coefficients<P>(pde);

  term_set<P> const &terms_vec_vec = pde.get_terms();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  element_table table(
      make_options({"-l", std::to_string(dimensions[0].get_level())}),
      pde.num_dims);

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
        partial_term<P> const &p_term = partial_terms[p_num];
        if (p_term.left_homo == homogeneity::inhomogeneous)
        {
          assert(static_cast<int>(p_term.left_bc_funcs.size()) > dim_num);

          fk::vector<P> const left_bc =
              boundary_conditions::compute_left_boundary_condition(
                  p_term.g_func, time, d, p_term.left_bc_funcs[dim_num]);

          /* compare to gold left bc */
          std::string const gold_filename =
              gold_filename_prefix + "bcL" + "_d" +
              std::to_string(d.get_degree()) + "_l" +
              std::to_string(d.get_level()) + "_t" + std::to_string(term_num) +
              "_d" + std::to_string(dim_num) + "_p" + std::to_string(p_num) +
              ".dat";

          fk::vector<P> const gold_left_bc_vector =
              fk::vector<P>(read_vector_from_txt_file(gold_filename));

          relaxed_comparison(gold_left_bc_vector, left_bc);
        }

        if (p_term.right_homo == homogeneity::inhomogeneous)
        {
          assert(static_cast<int>(p_term.right_bc_funcs.size()) > dim_num);

          fk::vector<P> const right_bc =
              boundary_conditions::compute_right_boundary_condition(
                  p_term.g_func, time, d, p_term.right_bc_funcs[dim_num]);

          /* compare to gold left bc */
          std::string const gold_filename =
              gold_filename_prefix + "bcR" + "_d" +
              std::to_string(d.get_degree()) + "_l" +
              std::to_string(d.get_level()) + "_t" + std::to_string(term_num) +
              "_d" + std::to_string(dim_num) + "_p" + std::to_string(p_num) +
              ".dat";

          fk::vector<P> const gold_right_bc_vector =
              fk::vector<P>(read_vector_from_txt_file(gold_filename));

          relaxed_comparison(gold_right_bc_vector, right_bc, eps_multiplier);
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

    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);

    generate_all_coefficients<TestType>(*pde);

    /* initialize bc vector at test_time */
    TestType const test_time = 5;

    std::vector<std::vector<std::vector<fk::vector<TestType>>>> left_bc_parts_1;
    std::vector<std::vector<std::vector<fk::vector<TestType>>>>
        right_bc_parts_1;
    int const start_element = 0;
    int const stop_element  = table.size() - 1;
    boundary_conditions::init_bc_parts(*pde, table, start_element, stop_element,
                                       left_bc_parts_1, right_bc_parts_1,
                                       test_time);

    fk::vector<TestType> bc_advanced_1 = boundary_conditions::generate_bc(
        left_bc_parts_1, right_bc_parts_1, *pde, start_element, stop_element,
        test_time);

    std::vector<std::vector<std::vector<fk::vector<TestType>>>> left_bc_parts_0;
    std::vector<std::vector<std::vector<fk::vector<TestType>>>>
        right_bc_parts_0;
    boundary_conditions::init_bc_parts(*pde, table, start_element, stop_element,
                                       left_bc_parts_0, right_bc_parts_0);

    fk::vector<TestType> bc_advanced_0 = boundary_conditions::generate_bc(
        left_bc_parts_0, right_bc_parts_0, *pde, start_element, stop_element,
        test_time);

    relaxed_comparison(bc_advanced_0, bc_advanced_1, 1);
  }

  /* Intead of calculating the entire boundary condition vector, calculate a
   * portion */
  SECTION("element table split")
  {
    /* setup stuff */
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);

    generate_all_coefficients<TestType>(*pde);

    /* initialize bc vector at test_time */
    TestType const test_time = 0;

    std::vector<std::vector<std::vector<fk::vector<TestType>>>> left_bc_parts_0;
    std::vector<std::vector<std::vector<fk::vector<TestType>>>>
        right_bc_parts_0;
    int const start_element_0 = 0;
    int const stop_element_0  = table.size() - 1;
    boundary_conditions::init_bc_parts(*pde, table, start_element_0,
                                       stop_element_0, left_bc_parts_0,
                                       right_bc_parts_0, test_time);

    fk::vector<TestType> bc_init = boundary_conditions::generate_bc(
        left_bc_parts_0, right_bc_parts_0, *pde, start_element_0,
        stop_element_0, test_time);

    /* create a vector for the first half of that vector */
    int index = 0;
    for (int table_element = 0; table_element < table.size(); ++table_element)
    {
      std::vector<std::vector<std::vector<fk::vector<TestType>>>> left_bc_parts;
      std::vector<std::vector<std::vector<fk::vector<TestType>>>>
          right_bc_parts;
      boundary_conditions::init_bc_parts(*pde, table, table_element,
                                         table_element, left_bc_parts,
                                         right_bc_parts);

      fk::vector<TestType> bc_advanced = boundary_conditions::generate_bc(
          left_bc_parts, right_bc_parts, *pde, table_element, table_element,
          test_time);

      fk::vector<TestType, mem_type::const_view> const bc_section(
          bc_init, index, index + bc_advanced.size() - 1);

      REQUIRE(bc_section == bc_advanced);

      index += bc_advanced.size();
    }
  }
}

TEMPLATE_TEST_CASE("compute_boundary_conditions", "[boundary_condition]",
                   double, float)
{
  double tol_factor = 0;
  if constexpr (std::is_same<TestType, float>::value)
    tol_factor = 1e1;
  else if constexpr (std::is_same<TestType, double>::value)
    tol_factor = 1e1;

  SECTION("diffusion_1 level 2 degree 2")
  {
    int const level  = 2;
    int const degree = 2;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "compute_boundary_conditions/"
                                             "diffusion1/";

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 4")
  {
    int const level  = 4;
    int const degree = 4;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "compute_boundary_conditions/"
                                             "diffusion1/";

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "compute_boundary_conditions/"
                                             "diffusion1/";

    test_compute_boundary_condition(*pde, gold_filename_prefix, tol_factor);
  }
}

TEMPLATE_TEST_CASE("boundary_conditions_vector", "[boundary_condition]", double,
                   float)
{
  double tol_factor = 0;
  if constexpr (std::is_same<TestType, float>::value)
    tol_factor = 1e4;
  else if constexpr (std::is_same<TestType, double>::value)
    tol_factor = 1e6;

  SECTION("diffusion_1 level 2 degree 2")
  {
    int const level  = 2;
    int const degree = 2;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "boundary_condition_vector/"
                                             "diffusion1/";

    test_boundary_condition_vector(*pde, gold_filename_prefix, tol_factor);
  }

  SECTION("diffusion_1 level 4 degree 4")
  {
    int const level  = 4;
    int const degree = 4;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "boundary_condition_vector/"
                                             "diffusion1/";

    test_boundary_condition_vector(*pde, gold_filename_prefix, tol_factor);
  }
  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde   = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix = "../testing/generated-inputs/"
                                             "boundary_condition_vector/"
                                             "diffusion1/";

    test_boundary_condition_vector(*pde, gold_filename_prefix, tol_factor);
  }
}
