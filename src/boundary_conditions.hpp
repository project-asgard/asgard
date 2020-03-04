#pragma once
#include "pde/pde_base.hpp"
#include "element_table.hpp"
#include "transformations.hpp"
#include "matlab_utilities.hpp"
#include "fast_math.hpp"

template< typename P >
class bc_timestepper
{
  public:

    bc_timestepper( PDE< P > const &pde,
                    element_table const &table, 
                    int const start_element,
                    int const stop_element,
                    P const t_init = 0 );

    fk::vector< P > advance( P const time ) const;

    void print_left() const;

    void print_right() const;

  private:

    std::vector< std::vector< std::vector< fk::vector< P > > > > left;

    std::vector< std::vector< std::vector< fk::vector< P > > > > right;

    P const t_init;

    int const bc_size;

    PDE< P > const &pde;

    void print( std::vector< std::vector< std::vector< fk::vector< P > > > > const &v ) const;
};

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
                            scalar_func<P> const time_func );

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
                                  scalar_func<P> const time_func );

template< typename P >
fk::vector< P >
boundary_condition_vector( PDE< P > const &pde,
                           element_table const &table,
                           P const time );

template< typename P >
std::vector< fk::vector< P > > 
generate_partial_bcs( PDE< P > const &pde,
                      std::vector< dimension< P > > const &dimensions,
                      int const d_index,
                      std::vector< vector_func< P > > const &bc_funcs, 
                      P const time,
                      std::vector< partial_term< P > > const &partial_terms,
                      int const p_index,
                      fk::vector< P > &&trace_bc );
