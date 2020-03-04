#include "tests_general.hpp"
#include "boundary_conditions.hpp"
#include "element_table.hpp"
#include "coefficients.hpp"

template< typename P >
void test_boundary_condition_vector( PDE< P > &pde, std::string gold_filename_prefix )
{
  /* setup stuff */
  dimension<P> const &d = pde.get_dimensions()[0];
  int const level       = d.get_level();

  element_table table(make_options({"-l", std::to_string(level)}),
                      pde.num_dims);

  generate_all_coefficients< P >( pde );

  /* initialize bc vector at test_time */
  P const test_time = 0;

  fk::vector< P > bc_init =
  boundary_condition_vector< P >( pde, table, test_time );

  bc_timestepper< P > bc_generator( pde, table, 0, table.size() - 1 );

  fk::vector< P > bc_advanced = bc_generator.advance( test_time );

  std::string const gold_filename = 
  gold_filename_prefix + "boundary_condition_vector.dat";

  fk::vector<P> const gold_bc_vector =
  fk::vector<P>(read_vector_from_txt_file(gold_filename));

  relaxed_comparison( gold_bc_vector, bc_advanced, 3e5 );

  return;
}

template< typename P >
void test_compute_boundary_condition( PDE<P> const &pde, std::string gold_filename_prefix )
{
  /* Open the file and feed params to compute_boundary_conditions */
  term_set< P > const &terms_vec_vec = pde.get_terms();

  std::vector< dimension< P > > const &dimensions = pde.get_dimensions();

  /* this timestep value must be consistent with the value used in the gold data generation
     scripts in matlab */
  P const time = 0;

  for( int term_num = 0; term_num < terms_vec_vec.size(); ++term_num )
  {
    std::vector< term< P > > const &terms_vec = terms_vec_vec[ term_num ];
    for( int dim_num = 0; dim_num < dimensions.size(); ++dim_num )
    {
      dimension< P > const &d = dimensions[ dim_num ];

      term< P > const &t = terms_vec[ dim_num ];

      std::vector< partial_term< P > > const &partial_terms = t.get_partial_terms();

      for( int p_num = 0; p_num < partial_terms.size(); ++p_num )
      {
        partial_term< P > const &p_term = partial_terms[ p_num ];
        if( p_term.left_homo == homogeneity::inhomogeneous )
        {
          assert( p_term.left_bc_funcs.size() > dim_num );
          fk::vector< P > const left_bc =
          compute_left_boundary_condition( pde,
                                           p_term.g_func,
                                           time,
                                           d.get_level(),
                                           d.get_degree(),
                                           d.domain_min,
                                           d.domain_max,
                                           p_term.left_bc_funcs[ dim_num ],
                                           p_term.left_bc_time_func);

          /* compare to gold left bc */
          std::string const gold_filename = 
          gold_filename_prefix +
          "bcL_t" + std::to_string( term_num ) + 
          "_d" + std::to_string( dim_num ) + 
          "_p" + std::to_string( p_num ) + 
          ".dat";

          fk::vector<P> const gold_left_bc_vector =
          fk::vector<P>(read_vector_from_txt_file(gold_filename));

          relaxed_comparison( gold_left_bc_vector, left_bc );
        }

        if( p_term.right_homo == homogeneity::inhomogeneous )
        {
          assert( p_term.right_bc_funcs.size() > dim_num );
          fk::vector< P > const right_bc =
          compute_right_boundary_condition( pde,
                                            p_term.g_func,
                                            time,
                                            d.get_level(),
                                            d.get_degree(),
                                            d.domain_min,
                                            d.domain_max,
                                            p_term.right_bc_funcs[ dim_num ],
                                            p_term.right_bc_time_func);
          /* compare to gold left bc */
          std::string const gold_filename = 
          gold_filename_prefix +
          "bcR_t" + std::to_string( term_num ) + 
          "_d" + std::to_string( dim_num ) + 
          "_p" + std::to_string( p_num ) + 
          ".dat";

          fk::vector<P> const gold_right_bc_vector =
          fk::vector<P>(read_vector_from_txt_file(gold_filename));

          relaxed_comparison( gold_right_bc_vector, right_bc );
        }
      }
    }
  }

  return;
}

TEMPLATE_TEST_CASE("v0 --> v1", "", double, float)
{
  SECTION( "time split" )
  {
    /* setup stuff */
    int const level  = 5;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);

    generate_all_coefficients< TestType >( *pde );

    /* initialize bc vector at test_time */
    TestType const test_time = 0;

    fk::vector< TestType > bc_init =
    boundary_condition_vector< TestType >( *pde, table, test_time );

    bc_timestepper< TestType > bc_generator( *pde, table, 0, table.size() - 1 );

    fk::vector< TestType > bc_advanced = bc_generator.advance( test_time );

    /* Captain! Same problem here */
    relaxed_comparison( bc_init, bc_advanced, 1 );
  }

  SECTION( "element table split" )
  {
    /* setup stuff */
    int const level  = 5;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    element_table table(make_options({"-l", std::to_string(level)}),
                        pde->num_dims);

    generate_all_coefficients< TestType >( *pde );

    /* initialize bc vector at test_time */
    TestType const test_time = 0;

    fk::vector< TestType > bc_init =
    boundary_condition_vector< TestType >( *pde, table, test_time );

    /* create a vector for the first half of that vector */
    int index = 0;
    for( int table_element = 0; table_element < table.size(); ++table_element )
    {
      bc_timestepper< TestType > bc_generator( *pde, table, table_element, table_element );
      fk::vector< TestType > bc_advanced = bc_generator.advance( test_time );
      
      fk::vector< TestType, mem_type::view > const 
      bc_section( bc_init, index, index + bc_advanced.size() - 1 );

      /* Captain! This passes because everything is zero */
      REQUIRE( bc_section == bc_advanced );

      index += bc_advanced.size();
    }
  }
}

TEMPLATE_TEST_CASE("compute_boundary_conditions", "null category", double, float)
{
  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix =
        "../testing/generated-inputs/"
        "compute_boundary_conditions/"
        "diffusion1/";

    test_compute_boundary_condition( *pde, gold_filename_prefix );
  }
}

TEMPLATE_TEST_CASE( "boundary_conditions_vector", "null category", double, float )
{
  SECTION("diffusion_1 level 5 degree 5")
  {
    int const level  = 5;
    int const degree = 5;
    auto const pde = make_PDE<TestType>(PDE_opts::diffusion_1, level, degree);

    std::string const gold_filename_prefix =
        "../testing/generated-inputs/"
        "boundary_condition_vector/"
        "diffusion1/";

    test_boundary_condition_vector( *pde, gold_filename_prefix );
  }
}
