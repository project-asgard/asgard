#pragma once
#include "asgard_pde_functions.hpp"
#include "asgard_wavelet_basis.hpp"

namespace asgard
{
template<typename P>
std::array<fk::matrix<P>, 4> generate_multi_wavelets(int const degree);

// used for testing of the forward-inverse transfrom, not used in the code
template<typename P>
fk::matrix<P> operator_two_scale(int const degree, int const num_levels);

} // namespace asgard
