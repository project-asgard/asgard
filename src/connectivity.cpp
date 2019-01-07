#include "connectivity.hpp"

#include <cmath>
#include "tensors.hpp"
#include <algorithm>
#include <numeric>

// Given a cell and level coordinate, return a 1-dimensional index
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
  std::fill(grid.begin(), grid.end(), 0);

  for (auto level = 0; level <= num_levels; ++level)
  {
    int const cell_boundary =
        static_cast<int>(std::pow(2, std::max(0, level - 1))) - 1;
    for (auto cell = 0; cell <= cell_boundary; ++cell)
    {
      int const i           = get_1d_index(level, cell);
      int const other_start = std::max(cell - 1, 0);
      int const other_end   = std::min(cell + 1, cell_boundary);
      fk::vector<int> other_cells(other_end - other_start + 1);
      std::iota(other_cells.begin(), other_cells.end(), other_start);
      std::transform(
          other_cells.begin(), other_cells.end(), other_cells.begin(),
          [level](int &other_cell) { return get_1d_index(level, other_cell); });

      // connect diagonal
      for (int const &j : other_cells)
      {
        grid(i, j) = 1;
        grid(j, i) = 1;
      }

      // connect periodic boundary
      if (cell == 0)
      {
        int const end = get_1d_index(level, cell_boundary);
        grid(i, end)  = 1;
        grid(end, i)  = 1;
      }

      if (cell == cell_boundary)
      {
        int const begin = get_1d_index(level, 0);
        grid(i, begin)  = 1;
        grid(begin, i)  = 1;
      }

      for (auto other_level = level + 1; other_level <= num_levels;
           ++other_level)
      {
        int const level_difference = [level, other_level]() {
          int difference = other_level - level;
          if (level == 0)
          {
            difference--;
          }
          return difference;
        }();

        int const diff_squared =
            static_cast<int>(std::pow(2, level_difference));
        int const other_boundary =
            static_cast<int>(std::pow(2, std::max(0, other_level - 1))) - 1;

        int const other_start = std::max((diff_squared * cell) - 1, 0);
        int const other_end =
            std::min(diff_squared * cell + diff_squared, other_boundary);
        //+3, leaving room for boundaries
        fk::vector<int> other_cells(other_end - other_start + 3);
        other_cells(0)                      = 0;
        other_cells(other_cells.size() - 1) = other_boundary;
        std::iota(other_cells.begin() + 1, other_cells.end() - 1, other_start);
        std::transform(other_cells.begin(), other_cells.end(),
                       other_cells.begin(), [other_level](int &other_cell) {
                         return get_1d_index(other_level, other_cell);
                       });

        for (int const &j : other_cells)
        {
          grid(i, j) = 1;
          grid(j, i) = 1;
        }
      }
    }
  }
  return grid;
}
