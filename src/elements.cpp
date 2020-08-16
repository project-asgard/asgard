#include "elements.hpp"

#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace elements
{
int64_t get_1d_index(int const level, int const cell)
{
  assert(level >= 0);
  assert(cell >= 0);

  if (level == 0)
  {
    return 0;
  }
  return static_cast<int64_t>(std::pow(2, level - 1)) + cell;
}

std::array<int64_t, 2> get_level_cell(int64_t const single_dim_id)
{
  assert(single_dim_id >= 0);
  if (single_dim_id == 0)
  {
    return {0, 0};
  }
  auto const level =
      static_cast<int64_t>(std::floor(std::log2(single_dim_id)) + 1);
  auto const cell =
      (level == 0) ? 0 : (single_dim_id - fm::two_raised_to(level - 1));
  return {level, cell};
}

template<typename P>
int64_t map_to_index(fk::vector<int> const &coords, options const &opts,
                     PDE<P> const &pde)
{
  assert(coords.size() == pde.num_dims * 2);

  int64_t id     = 0;
  int64_t stride = 1;

  for (auto i = 0; i < pde.num_dims; ++i)
  {
    assert(coords(i) >= 0);
    assert(coords(i) <= opts.max_level);
    assert(coords(i + pde.num_dims) >= 0);

    id += get_1d_index(coords(i), coords(i + pde.num_dims)) * stride;
    stride *= static_cast<int64_t>(std::pow(2, opts.max_level));
  }

  assert(id >= 0);
  assert(id <=
         static_cast<int64_t>(std::pow(2, opts.max_level * pde.num_dims)));
  return id;
}

template<typename P>
fk::vector<int>
map_to_coords(int64_t const id, options const &opts, PDE<P> const &pde)
{
  assert(id >= 0);

  auto const stride = static_cast<int64_t>(std::pow(2, opts.max_level));

  fk::vector<int> coords(pde.num_dims * 2);
  for (auto i = 0; i < pde.num_dims; ++i)
  {
    auto const id_1d = static_cast<int64_t>(
        std::round((id / static_cast<int64_t>(std::pow(stride, i))) % stride));
    auto const [lev, pos]    = get_level_cell(id_1d);
    coords(i)                = lev;
    coords(i + pde.num_dims) = pos;
  }
  return coords;
}

// construct element table
template<typename P>
table::table(options const opts, PDE<P> const &pde)
{
  auto const perm_table = [&pde, &opts]() {
    auto const sort = false;

    auto const dims = pde.get_dimensions();
    fk::vector<int> levels(pde.num_dims);
    std::transform(dims.begin(), dims.end(), levels.begin(),
                   [](auto const &dim) { return dim.get_level(); });

    return opts.use_full_grid
               ? permutations::get_max_multi(levels, pde.num_dims, sort)
               : permutations::get_lequal_multi(
                     levels, pde.num_dims,
                     *std::max_element(levels.begin(), levels.end()), sort);
  }();

  fk::vector<int> dev_table_builder;
  for (int row = 0; row < perm_table.nrows(); ++row)
  {
    // get the level tuple to work on
    fk::vector<int> const level_tuple =
        perm_table.extract_submatrix(row, 0, 1, pde.num_dims);
    // calculate all possible cell indices allowed by this level tuple
    fk::matrix<int> const index_set = get_cell_index_set(level_tuple);

    for (int cell_set = 0; cell_set < index_set.nrows(); ++cell_set)
    {
      auto const cell_indices = fk::vector<int>(
          index_set.extract_submatrix(cell_set, 0, 1, pde.num_dims));

      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      auto const coords = fk::vector<int>(level_tuple).concat(cell_indices);
      auto const key    = map_to_index(coords, opts, pde);

      active_element_ids_.push_back(key);

      id_to_coords_[key].resize(coords.size()) = coords;

      // assign into flattened device table builder
      dev_table_builder.concat(coords);
    }
  }

  assert(active_element_ids_.size() == id_to_coords_.size());
  active_table_d_.resize(dev_table_builder.size())
      .transfer_from(dev_table_builder);
}

// static construction helper
// return the cell indices, given a level tuple
// each row in the returned matrix is the cell portion of an element's
// coordinate
fk::matrix<int> table::get_cell_index_set(fk::vector<int> const &level_tuple)
{
  assert(level_tuple.size() > 0);
  for (auto const level : level_tuple)
  {
    assert(level >= 0);
  }
  int const num_dims = level_tuple.size();

  // get number of cells for each level coordinate in the input tuple
  // 2^(max(0, level-1))
  fk::vector<int> const cells_per_level = [level_tuple]() {
    fk::vector<int> v(level_tuple.size());
    std::transform(
        level_tuple.begin(), level_tuple.end(), v.begin(),
        [](int level) { return fm::two_raised_to(std::max(0, level - 1)); });
    return v;
  }();

  // total number of cells for an entire level tuple will be all possible
  // combinations, so just a product of the number of cells per level in that
  // tuple
  int const total_cells = [cells_per_level]() {
    int total = 1;
    return std::accumulate(cells_per_level.begin(), cells_per_level.end(),
                           total, std::multiplies<int>());
  }();

  // allocate the returned
  fk::matrix<int> cell_index_set(total_cells, num_dims);

  // recursion base case
  if (num_dims == 1)
  {
    std::vector<int> entries(total_cells);
    std::iota(begin(entries), end(entries), 0);
    cell_index_set.update_col(0, entries);
    return cell_index_set;
  }

  // recursively build the cell index set
  int const cells_this_dim = cells_per_level(num_dims - 1);
  int const rows_per_iter  = total_cells / cells_this_dim;
  for (auto i = 0; i < cells_this_dim; ++i)
  {
    int const row_pos = i * rows_per_iter;
    fk::matrix<int> partial_result(rows_per_iter, num_dims);
    fk::vector<int> partial_level_tuple = level_tuple;
    partial_level_tuple.resize(num_dims - 1);
    partial_result.set_submatrix(0, 0, get_cell_index_set(partial_level_tuple));
    std::vector<int> last_col(rows_per_iter, i);
    partial_result.update_col(num_dims - 1, last_col);
    cell_index_set.set_submatrix(row_pos, 0, partial_result);
  }

  return cell_index_set;
}

template int64_t map_to_index(fk::vector<int> const &coords,
                              options const &opts, PDE<float> const &pde);

template int64_t map_to_index(fk::vector<int> const &coords,
                              options const &opts, PDE<double> const &pde);

template fk::vector<int>
map_to_coords(int64_t const id, options const &opts, PDE<float> const &pde);

template fk::vector<int>
map_to_coords(int64_t const id, options const &opts, PDE<double> const &pde);

template table::table(options const program_opts, PDE<float> const &pde);

template table::table(options const program_opts, PDE<double> const &pde);

} // namespace elements
