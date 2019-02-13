#include "coefficients.hpp"
#include "pde.hpp"
#include "tensors.hpp"

template<typename P>
fk::matrix<P>
generate_coefficients(dimension<P> const dim,
                      std::vector<term<P>> const term_list, P const time)
{
  return fk::matrix<P>();
}

template fk::matrix<float>
generate_coefficients(dimension<float> const dim,
                      std::vector<term<float>> const term_list,
                      float const time);

template fk::matrix<double>
generate_coefficients(dimension<double> const dim,
                      std::vector<term<double>> const term_list,
                      double const time);
