#pragma once

#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree);

template<typename P>
fk::matrix<P> operator_two_scale(int const degree, int const num_levels);

template<typename P>
fk::matrix<P> apply_left_fmwt(fk::matrix<P> const &fmwt,
                              fk::matrix<P> const &coefficient_matrix,
                              int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P>
apply_left_fmwt_transposed(fk::matrix<P> const &fmwt,
                           fk::matrix<P> const &coefficient_matrix,
                           int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P> apply_right_fmwt(fk::matrix<P> const &fmwt,
                               fk::matrix<P> const &coefficient_matrix,
                               int const kdeg, int const num_levels);
template<typename P>
fk::matrix<P>
apply_right_fmwt_transposed(fk::matrix<P> const &fmwt,
                            fk::matrix<P> const &coefficient_matrix,
                            int const kdeg, int const num_levels);

// FIXME add above to namespace
namespace basis
{
template<typename P>
class wavelet_transform
{
public:
  wavelet_transform(int const max_level, int const degree);

  // given a level, and the row/col number, retrieve the value occupying
  // position (row, col) in the transform operator for that level.
  P get_value(int const level, int const i, int const j) const;

  // exposed for testing
  std::vector<fk::matrix<P>> const &get_blocks() const { return dense_blocks_; }

private:
  // dense regions of the transform operator. store all for every level up to
  // max_level. each level contains all dense_blocks_ from lower levels, except
  // for one level-unique block. blocks are stored as (unique blocks for levels
  // 0 -> max_level | max level blocks)
  std::vector<fk::matrix<P>> dense_blocks_; // FIXME may eventually change to
                                            // vector of views of larger matrix
  int const max_level_;
};

} // namespace basis
