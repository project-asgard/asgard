#pragma once
#include "pde.hpp"
#include "tensors.hpp"

template< typename P >
void generate_all_coefficients( std::unique_ptr< PDE< P > > &pde,
                                double const time = 0.0,
                                bool const rotate = true );

extern template
void generate_all_coefficients< float >( std::unique_ptr< PDE< float > > &pde,
                                         double const time = 0.0,
                                         bool const rotate = true );

extern template
void generate_all_coefficients< double >( std::unique_ptr< PDE< double > > &pde,
                                          double const time = 0.0,
                                          bool const rotate = true );

template<typename P>
fk::matrix< double >
generate_coefficients(dimension<P> const &dim,
                      term<P> const &term_1D, 
                      partial_term< P > const &pterm,
                      double const time = 0.0,
                      bool const rotate = true);

extern template
fk::matrix< double >
generate_coefficients< float >( dimension<float> const &dim,
                                term<float> const &term_1D,
                                partial_term< float > const &pterm,
                                double const time = 0.0,
                                bool const rotate = true);

extern template
fk::matrix< double >
generate_coefficients< double >( dimension<double> const &dim,
                                 term<double> const &term_1D,
                                 partial_term< double > const &pterm,
                                 double const time = 0.0,
                                 bool const rotate = true);
