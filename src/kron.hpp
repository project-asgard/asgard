#include "tensors.hpp"
#include "fast_math.hpp"
#include "transformations.hpp"
#include <iostream>

template< typename P >
fk::vector< P, mem_type::owner >
kron( std::vector< fk::matrix< P, mem_type::owner > > const &matrix, 
      fk::vector< P, mem_type::owner > const &x );

template< typename P >
int calculate_workspace_len( std::vector< fk::matrix< P, mem_type::owner > > const &matrix,
                             int const x_size );

extern template
fk::vector< double, mem_type::owner >
kron( std::vector< fk::matrix< double, mem_type::owner > > const &matrix, 
      fk::vector< double, mem_type::owner > const &x );

extern template
fk::vector< float, mem_type::owner >
kron( std::vector< fk::matrix< float, mem_type::owner > > const &matrix, 
      fk::vector< float, mem_type::owner > const &x );

extern template
int calculate_workspace_len( std::vector< fk::matrix< float, mem_type::owner > > 
                             const &matrix, int const x_size );

extern template
int calculate_workspace_len( std::vector< fk::matrix< double, mem_type::owner > > 
                             const &matrix, int const x_size );
