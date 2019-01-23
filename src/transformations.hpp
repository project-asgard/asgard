#pragma once

#include "element_table.hpp"
#include "tensors.hpp"

#include <vector>

template<typename P>
class multi_wavelets
{
  fk::matrix<P> phi_co;
  fk::matrix<P> scalet_coefficients;
  fk::matrix<P> g0;
  fk::matrix<P> g1;
  fk::matrix<P> h0;
  fk::matrix<P> h1;

public:
  multi_wavelets(int const degree);
  fk::matrix<P> get_h0() const;
  fk::matrix<P> get_g0() const;
  fk::matrix<P> get_h1() const;
  fk::matrix<P> get_g1() const;
  fk::matrix<P> get_phi_co() const;
  fk::matrix<P> get_scalet_coefficients() const;
};

extern template class multi_wavelets<double>;
extern template class multi_wavelets<float>;

template<typename P>
fk::vector<P>
combine_dimensions(Options const, element_table const &,
                   std::vector<fk::vector<P> const> const &, P const);

extern template fk::vector<double>
combine_dimensions(Options const, element_table const &,
                   std::vector<fk::vector<double> const> const &, double const);
extern template fk::vector<float>
combine_dimensions(Options const, element_table const &,
                   std::vector<fk::vector<float> const> const &, float const);
