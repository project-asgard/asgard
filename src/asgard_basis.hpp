#pragma once
#include "asgard_block_matrix.hpp"
#include "asgard_pde.hpp"
#include "asgard_wavelet_basis.hpp"

namespace asgard
{
template<typename P>
std::array<fk::matrix<P>, 4> generate_multi_wavelets(int const degree);

// used for testing of the forward-inverse transfrom, not used in the code
template<typename P>
fk::matrix<P> operator_two_scale(int const degree, int const num_levels);

// FIXME add above to namespace
namespace basis
{
enum class side
{
  left,
  right
};
enum class transpose
{
  no_trans,
  trans
};

template<typename P, resource resrc>
class wavelet_transform
{
public:
  wavelet_transform(int const max_level_in, int const max_degree,
                    verbosity_level verb = verbosity_level::quiet);

  wavelet_transform(PDE<P> const &pde, verbosity_level verb = verbosity_level::quiet)
      : // Choose which max level to use depending on whether adaptivity is
        // enabled.
        // If adaptivity is enabled, use the program_opts.max_level
        // Otherwise, use the max_level defined in the PDE (max level of all
        // dims)
        wavelet_transform(pde.max_level(),
                          pde.get_dimensions()[0].get_degree(), verb)
  {}

  // apply the fmwt matrix to coefficients
  template<mem_type omem>
  fk::matrix<P, mem_type::owner, resrc>
  apply(fk::matrix<P, omem, resrc> const &coefficients, int const level,
        basis::side const transform_side,
        basis::transpose const transform_trans) const;

  // shim to apply fmwt to vectors
  template<mem_type omem>
  fk::vector<P, mem_type::owner, resrc>
  apply(fk::vector<P, omem, resrc> const &coefficients, int const level,
        basis::side const transform_side,
        basis::transpose const transform_trans) const;

  // exposed for testing
  std::vector<fk::matrix<P, mem_type::owner, resrc>> const &get_blocks() const
  {
    return dense_blocks_;
  }

  int const max_level;
  int const degree;

private:
  // dense regions of the transform operator. store all for every level up to
  // max_level. each level contains all dense_blocks_ from lower levels, except
  // for one level-unique block. blocks are stored as (unique blocks for levels
  // 0 -> max_level | max level blocks)
  std::vector<fk::matrix<P, mem_type::owner, resrc>>
      dense_blocks_; // FIXME may eventually change to
                     // vector of views of larger matrix
};

} // namespace basis
} // namespace asgard
