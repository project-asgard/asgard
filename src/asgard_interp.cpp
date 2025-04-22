#include "asgard_interp.hpp"

namespace asgard
{

template<typename P, int degree>
void interpolation_manager1d<P, degree>::initialize_nodes(int const max_level)
{
  int const num_cells = fm::ipow2(max_level);

  nodes_ = vector2d<P>(n, num_cells);

  // starting numerator for the counting process
  P constexpr start_num = (degree % 2 == 1) ? 1 : 0;

  // for degree 1, 2, 3 ..., start start_den is 3, 3, 5, 5 ...
  P constexpr start_den = 1 + degree + ((degree % 2 == 1) ? 1 : 0);

  // doing level 0
  P den = start_den;
  for (int i = 0; i < n; i++)
    nodes_[0][i] = (start_num + i) / start_den;

  // follow on levels follow a pattern of jumps in the numerators
  std::array<P, n> jumps = []() -> std::array<P, n> {
      if constexpr (degree == 1)
        return {4, 2};
      else if constexpr (degree == 2)
        return {2, 2, 2};
      else // if constexpr (degree == 3) {
        return {2, 4, 2, 2};
    }();

  for (int l = 1; l <= max_level; l++) {
    den *= 2;

    P num = start_num + ((degree % 2 == 1) ? 0 : 1);

    int const cell_start = fm::ipow2(l - 1);
    int const cell_end   = 2 * cell_start;
    for (int c = cell_start; c < cell_end; c++) {
      for (int i = 0; i < n; i++) {
        nodes_[c][i] = num / den;
        num += jumps[i];
      }
    }
  }
}


#ifdef ASGARD_ENABLE_DOUBLE
template class interpolation_manager1d<double, 1>;
template class interpolation_manager1d<double, 2>;
template class interpolation_manager1d<double, 3>;
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class interpolation_manager1d<float, 1>;
template class interpolation_manager1d<float, 2>;
template class interpolation_manager1d<float, 3>;
#endif

} // namespace asgard
