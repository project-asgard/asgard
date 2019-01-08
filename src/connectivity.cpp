#include "connectivity.hpp"


#include <cmath>
#include "matlab_utilities.hpp"
#include "permutations.hpp"
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
fk::matrix<int> make_1d_connectivity(int const num_levels)
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

// Generate connectivity for num_dims dimensions
//
// From MATLAB:
// This code is to generate the ndimensional connectivity...
// Here, we consider the maximum connectivity, which includes all overlapping
// cells, neighbor cells, and the periodic boundary cells
list_set make_connectivity(element_table table, int const num_dims,
                           int const max_level_sum, int const max_level_val,
                           bool const sort_connected)
{
  list_set connectivity;
  connectivity.reserve(table.size());

  // step 1: generate 1d connectivity
  int const num_levels             = std::max(max_level_sum, max_level_val);
  fk::matrix<int> const connect_1d = make_1d_connectivity(num_levels);
  std::vector<int> levels, cells;
  // step 2: 1d mesh, all possible combinations
  for (auto i = 0; i <= num_levels; ++i)
  {
    int const num_cells = static_cast<int>(std::pow(2, std::max(0, i - 1))) - 1;
    for (auto j = 0; j <= num_cells; ++j)
    {
      levels.push_back(i);
      cells.push_back(j);
    }
  }
  fk::matrix<int> mesh_1d(levels.size(), 2);
  mesh_1d.update_col(0, levels);
  mesh_1d.update_col(1, cells);

  // step 3: num_dims connectivity
  for (auto i = 0; i < table.size(); ++i)
  {
    fk::vector<int> const coords = table.get_coords(i);

    list_set levels_lists, cells_lists;
    // iterate over the cell portion of the coordinates...
    for (auto dim = num_dims; dim < coords.size(); ++dim)
    {
      int const cell_coord = coords(dim);
      fk::vector<int> const connect_row =
          connect_1d.extract_submatrix(cell_coord, 0, 1, connect_1d.ncols());
      fk::vector<int> const non_zeros =
          find(connect_row, [](int const &elem) { return elem != 0; });
      fk::vector<int> levels(non_zeros.size());
      fk::vector<int> cells(non_zeros.size());

      for (auto k = 0; k < non_zeros.size(); ++k)
      {
        levels(k) = mesh_1d(non_zeros(k), 0);
        cells(k)  = mesh_1d(non_zeros(k), 1);
      }
      levels_lists.push_back(levels);
      cells_lists.push_back(cells);
    }

    fk::matrix<int> const index_matrix = get_leq_max_indices(
        levels_lists, num_dims, max_level_sum, max_level_val);

    fk::vector<int> connected_elements(index_matrix.nrows());
    for (auto element = 0; element < index_matrix.nrows(); ++element)
    {
      fk::vector<int> key(num_dims * 2);
      for (auto dim = 0; dim < index_matrix.ncols(); ++dim)
      {
        int const level_coord = levels_lists[dim](index_matrix(element, dim));
        int const cell_coord  = cells_lists[dim](index_matrix(element, dim));
        int const level_pos   = dim;
        int const cell_pos    = dim + num_dims;
        key(level_pos)        = level_coord;
        key(cell_pos)         = cell_coord;
      }
      connected_elements(element) = table.get_index(key);
    }

    if (sort_connected)
    {
      std::sort(connected_elements.begin(), connected_elements.end(),
                std::less<int>());
    }
    connectivity.push_back(connected_elements);
  }

  return connectivity;
}
