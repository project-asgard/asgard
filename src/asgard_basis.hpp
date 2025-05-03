#pragma once
#include "asgard_pde_functions.hpp"
#include "asgard_wavelet_basis.hpp"

namespace asgard
{
template<typename P>
std::array<fk::matrix<P>, 4> generate_multi_wavelets(int const degree);

} // namespace asgard
