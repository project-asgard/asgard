#pragma once

#include "tensors.hpp"

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

// suppress implicit instantiations later on
extern template class multi_wavelets<double>;
extern template class multi_wavelets<float>;
