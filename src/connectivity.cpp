#include "connectivity.hpp"

#include <cmath>
#include "tensors.hpp"

// Given a cell and level coordinate, return a 1-dimensional index
// FIXME does this need to be public/exposed?
int get_1d_index(int const level, int const cell)
{
  assert(level >= 0);
  assert(cell >= 0);

  if (level == 0)
  {
    return 0;
  }
  return static_cast<int>(std::pow(2, level - 1)) + cell;
}

// Build connectivity for single dimension
fk::matrix<int> connect_1d(int const num_levels)
{
  assert(num_levels > 0);

  int const lev_squared = static_cast<int>(std::pow(2, num_levels));
  fk::matrix<int> grid(lev_squared, lev_squared);

  for (auto level = 0; level < num_levels; ++level)
  {
    int const num_cells = static_cast<int>(std::pow(2, std::max(0, level - 1)));
    for (auto cell = 0; cell < num_cells; ++cell)
    {
      int const i = get_1d_index(level, cell);
    }
  }
  return grid;
}
