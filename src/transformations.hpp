#pragma once

#include "element_table.hpp"
#include "tensors.hpp"

#include <vector>

template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree);

extern template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
extern template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

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
