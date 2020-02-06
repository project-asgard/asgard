#include "boundary_conditions.hpp"
#include <cstdio>

template< typename P >
std::vector< fk::vector< P > > 
generate_partial_bcs( PDE< P > const &pde,
                      std::vector< dimension< P > > const &dimensions,
                      int const d_index,
                      std::vector< vector_func< P > > const &bc_funcs, 
                      P const time,
                      std::vector< partial_term< P > > const &partial_terms,
                      int const p_index,
                      fk::vector< P > &&trace_bc )
{
  assert( d_index < dimensions.size() );

  std::vector< fk::vector< P > > partial_bc_vecs;

  for( int dim_num = 0; dim_num < d_index; ++dim_num )
  {
      partial_bc_vecs.emplace_back( forward_transform( dimensions[ dim_num ],
                                                       bc_funcs[dim_num],
                                                       time ) );
  }

  partial_bc_vecs.emplace_back( std::move( trace_bc ) );

  /* Convert basis operator from double to typename P */
  fm::gemv( fk::matrix< P >( dimensions[d_index].get_to_basis_operator() ), 
            fk::vector< P >( partial_bc_vecs.back() ), 
            partial_bc_vecs.back() );

  /* Captain! This will not run with the current tests */
  if( p_index > 0 )
  {
    fk::matrix< P > chain = eye< P >( partial_terms[ 0 ].get_coefficients().nrows(),
                                      partial_terms[ 0 ].get_coefficients().ncols() );

    for( int p = 0; p < p_index; ++p )
    {
      fm::gemm( fk::matrix< P >( chain ), partial_terms[ p ].get_coefficients(), chain );
    }

    fm::gemv( chain, fk::vector< P >( partial_bc_vecs.back() ), partial_bc_vecs.back() );
  }

  for( int dim_num = d_index + 1; dim_num < dimensions.size(); ++dim_num )
  {
      partial_bc_vecs.emplace_back( forward_transform( dimensions[ dim_num ],
                                                       bc_funcs[dim_num],
                                                       time ) );
  }

  /* Captain! Test code */
  /*
  std::printf( "partial vectors:\n");
  for( auto f : partial_bc_vecs )
  {
    std::printf( "partial bc:\n");
    for( int i = 0; i < f.size(); ++i )
    {
      std::printf("value: %d: %lf\n", i + 1, f( i ) );
    }
  }
  */
  /* End test code */

  return partial_bc_vecs;
}

template< typename P >
fk::vector< P >
boundary_condition_vector( PDE< P > const &pde,
                           element_table const &table,
                           P const time )
{
  /* Open the file and feed params to compute_boundary_conditions */
  term_set< P > const &terms_vec_vec = pde.get_terms();

  std::vector< dimension< P > > const &dimensions = pde.get_dimensions();

  fk::vector< P > bc( table.size() * 
                      std::pow( dimensions[ 0 ].get_degree(), dimensions.size() ) );

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
          fk::vector< P > trace_bc =
          compute_left_boundary_condition( pde,
                                           p_term.g_func,
                                           time,
                                           d.get_level(),
                                           d.get_degree(),
                                           d.domain_min,
                                           d.domain_max,
                                           p_term.left_bc_funcs[ dim_num ], 
                                           p_term.left_bc_time_func );

          std::vector< fk::vector< P > > p_term_left_bcs =
          generate_partial_bcs( pde, dimensions, dim_num , p_term.left_bc_funcs, time,
                                partial_terms, p_num, std::move( trace_bc ) );

          fk::vector< P > combined = combine_dimensions( d.get_degree(),
                                        table,
                                        0,
                                        table.size() - 1,
                                        p_term_left_bcs,
                                        p_term.left_bc_time_func( time ) );

          fm::axpy( combined, bc );
        }

        if( p_term.right_homo == homogeneity::inhomogeneous )
        {
          fk::vector< P > trace_bc =
          compute_right_boundary_condition(pde,
                                           p_term.g_func,
                                           time,
                                           d.get_level(),
                                           d.get_degree(),
                                           d.domain_min,
                                           d.domain_max,
                                           p_term.right_bc_funcs[ dim_num ], 
                                           p_term.right_bc_time_func );

          std::vector< fk::vector< P > > p_term_right_bcs =
          generate_partial_bcs( pde, dimensions, dim_num , p_term.right_bc_funcs, time,
                                partial_terms, p_num, std::move( trace_bc ) );

          fk::vector< P > combined = combine_dimensions( d.get_degree(),
                                        table,
                                        0,
                                        table.size() - 1,
                                        p_term_right_bcs,
                                        p_term.right_bc_time_func( time ) );

          fm::axpy( combined, bc );
        }
      }
    }
  }

  return bc;
}

template< typename P >
fk::vector< P >
compute_right_boundary_condition( PDE< P > const &pde,
                                  g_func_type const g_func,
                                  P const time,
                                  int level,
                                  int degree,
                                  P const domain_min,
                                  P const domain_max,
                                  vector_func< P > const bc_func,
                                  scalar_func<P> const time_func )
{
  P const domain_extent = domain_max - domain_min;
  assert( domain_extent > 0 );

  P const total_cells = std::pow( 2, level );

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = degree * total_cells;

  fk::vector< P > bc( dof );

  P g = g_func( domain_max, time );
  if( !std::isfinite( g ) )
  {
    P const small_dx = domain_per_cell * 1e-7;
    g = g_func( domain_max - small_dx, time );

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    assert( std::isfinite( g ) );
  }

  /* Captain! Use the new upcoming vector-view of matrix column constructor */
  fk::vector< P > legendre_polys_at_value =
  fk::vector< P >( legendre(fk::vector<P>{1}, degree, legendre_normalization::lin )[0] );

  P const scale_factor = 
  ( 1.0 / std::sqrt( domain_per_cell ) ) * 
  bc_func( fk::vector< P >( {domain_max} ), time )( 0 ) * g;

  legendre_polys_at_value.scale( scale_factor );

  int const start_index = degree * ( total_cells - 1 );
  int const stop_index = degree * total_cells - 1;
  fk::vector< P, mem_type::view > destination_slice( bc, start_index, stop_index );

  assert( destination_slice.size() == legendre_polys_at_value.size() );

  destination_slice = fk::vector< P >( legendre_polys_at_value );

  return bc;
}

template< typename P >
fk::vector< P >
compute_left_boundary_condition( PDE< P > const &pde,
                            g_func_type const g_func,
                            P const time,
                            int level,
                            int degree,
                            P const domain_min,
                            P const domain_max,
                            vector_func< P > const bc_func, 
                            scalar_func<P> const time_func )
{
  P const domain_extent = domain_max - domain_min;
  assert( domain_extent > 0 );

  P const total_cells = std::pow( 2, level );

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = degree * total_cells;

  fk::vector< P > bc( dof );

  P g = g_func( domain_min, time );
  if( !std::isfinite( g ) )
  {
    P const small_dx = domain_per_cell * 1e-7;
    g = g_func( domain_min + small_dx, time );

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    assert( std::isfinite( g ) );
  }

  P const scale_factor = 
  ( 1.0 / std::sqrt( domain_per_cell ) ) * 
  bc_func( fk::vector< P >( {domain_min} ), time )( 0 ) * g * -1.0;

  /* legendre() returns a 1D matrix - must be converted into a vector */
  fk::vector< P > legendre_polys_at_value =
  fk::vector< P >( legendre(fk::vector<P>{-1}, degree, legendre_normalization::lin )[0] );

  legendre_polys_at_value.scale( scale_factor );

  fk::vector< P, mem_type::view > destination_slice( bc, 0, degree - 1 );

  assert( destination_slice.size() == legendre_polys_at_value.size() );

  destination_slice = fk::vector< P >( legendre_polys_at_value );

  return bc;
}

/* explicit instantiations */
template
fk::vector< float >
compute_left_boundary_condition( PDE< float > const &pde,
                            g_func_type const g_func,
                            float const time,
                            int level,
                            int degree,
                            float const domain_min,
                            float const domain_max,
                            vector_func< float > const bc_func,
                            scalar_func< float > const time_func );

template
fk::vector< double >
compute_left_boundary_condition( PDE< double > const &pde,
                            g_func_type const g_func,
                            double const time,
                            int level,
                            int degree,
                            double const domain_min,
                            double  const domain_max,
                            vector_func< double > const bc_func,
                            scalar_func< double > const time_func );

template
fk::vector< float >
compute_right_boundary_condition( PDE< float > const &pde,
                            g_func_type const g_func,
                            float const time,
                            int level,
                            int degree,
                            float const domain_min,
                            float const domain_max,
                            vector_func< float > const bc_func,
                            scalar_func< float > const time_func );

template
fk::vector< double >
compute_right_boundary_condition( PDE< double > const &pde,
                            g_func_type const g_func,
                            double const time,
                            int level,
                            int degree,
                            double const domain_min,
                            double  const domain_max,
                            vector_func< double > const bc_func,
                            scalar_func< double > const time_func );

template
fk::vector< float >
boundary_condition_vector( PDE< float > const &pde,
                           element_table const &table,
                           float const time );

template
fk::vector< double >
boundary_condition_vector( PDE< double > const &pde,
                           element_table const &table,
                           double const time );
template
std::vector< fk::vector< double > > 
generate_partial_bcs( PDE< double > const &pde,
                      std::vector< dimension< double > > const &dimensions,
                      int const d_num,
                      std::vector< vector_func< double > > const &bc_funcs, 
                      double const time,
                      std::vector< partial_term< double > > const &partial_terms,
                      int const p_index,
                      fk::vector< double > &&trace_bc );

template
std::vector< fk::vector< float > > 
generate_partial_bcs( PDE< float > const &pde,
                      std::vector< dimension< float > > const &dimensions,
                      int const d_index,
                      std::vector< vector_func< float > > const &bc_funcs, 
                      float const time,
                      std::vector< partial_term< float > > const &partial_terms,
                      int const p_index,
                      fk::vector< float > &&trace_bc );
