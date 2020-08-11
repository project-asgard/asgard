#include "element_table.hpp"

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

template<typename P>
int64_t map_to_index(fk::vector<int> const &coords, options const &opts,
                     PDE<P> const &pde)
{
  assert(coords.size() * 2 == pde.num_dims);

  int64_t id     = 0;
  int64_t stride = 1;
  for (int i = 0; i < pde.num_dims; ++i)
  {
    assert(coords(i) < opts.max_level);
    id += get_1d_index(coords(i), coords(i + pde.num_dims)) * stride;
    stride += stride * fm::two_raised_to(opts.max_level);
  }
  assert(id >= 0);
  assert(id < fm::two_raised_to(static_cast<int64_t>(opts.max_level) *
                                pde.num_dims));
  return id;
}

template<typename P>
fk::vector<int>
map_to_coords(int64_t const id, options const &opts, PDE<P> const &pde)
{
  assert(id >= 0);

  return fk::vector<int>();
}

// construct forward and reverse element tables
element_table::element_table(options const program_opts, int const num_levels,
                             int const num_dims)
{
  bool const use_full_grid = program_opts.use_full_grid;

  assert(num_dims > 0);
  assert(num_levels > 1);

  // get permutation table for some num_dims, num_levels
  // each row of this table becomes a level tuple, and is the "level" component
  // of some number of elements' coordinates
  fk::matrix<int> const perm_table =
      use_full_grid ? permutations::get_max(num_dims, num_levels, false)
                    : permutations::get_lequal(num_dims, num_levels, false);

  // in the matlab, the tables sizes are precomputed/preallocated.
  // I wrote the code to do that, however, this isn't a performance
  // critical area (constructed only once at startup) and we'd need
  // to explore the thread-safety of our tables / build a thread-safe
  // table to see any benefit. -TM

  // build the element tables (forward and reverse) and setup device table
  int index = 0;

  fk::vector<int> dev_table_builder;
  for (int row = 0; row < perm_table.nrows(); ++row)
  {
    // get the level tuple to work on
    fk::vector<int> const level_tuple =
        perm_table.extract_submatrix(row, 0, 1, num_dims);
    // calculate all possible cell indices allowed by this level tuple
    fk::matrix<int> const index_set = get_cell_index_set(level_tuple);

    for (int cell_set = 0; cell_set < index_set.nrows(); ++cell_set)
    {
      fk::vector<int> cell_indices =
          index_set.extract_submatrix(cell_set, 0, 1, num_dims);

      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      fk::vector<int> key = level_tuple;
      key.concat(cell_indices);

      // assign into flattened device table builder
      dev_table_builder.concat(key);

      forward_table_[key] = index++;

      // note the matlab code has an option to append 1d cell indices to the
      // reverse element table. //FIXME do we need to precompute or can we call
      // the 1d helper as needed?
      reverse_table_.push_back(key);
    }
  }
  assert(forward_table_.size() == reverse_table_.size());
  reverse_table_d_.resize(dev_table_builder.size())
      .transfer_from(dev_table_builder);
}

// new version: construct forward and reverse element tables
// TODO need explicit instant. for this constructor
template<typename P>
element_table::element_table(options const opts, PDE<P> const &pde)
{
  // assert(iscolumn(lev_vec) || isrow(lev_vec));
  // num_dimensions = numel(lev_vec);

  //% num_dimensions = numel(pde.dimensions);
  // is_sparse_grid = strcmp( grid_type, 'SG');

  //%%
  //% Setup element table as a collection of sparse vectors to
  //% store the lev and cell info for each dim.

  //%%
  // calculate maximum refinement
  auto const elems_1d = fm::two_raised_to(static_cast<int64_t>(opts.max_level));
  auto const num_max_elems_fp = std::pow(elems_1d, pde.num_dims);
  assert(num_max_elems_fp < static_cast<double>(INT64_MAX));
  auto const num_max_elements = static_cast<int64_t>(num_max_elems_fp);

  //%%
  //% allocate the sparse element table members

  // elements.lev_p1     = sparse (num_elements_max, num_dimensions); % _p1 is
  // for "plus 1" sinse sparse cannot accpet 0 elements.pos_p1     = sparse
  // (num_elements_max, num_dimensions); elements.type       = sparse
  // (num_elements_max, 1);

  //%%
  //% Get combinations of elements across dimensions and apply sparse-grid
  // selection rule

  // get permutation table for some num_dims, num_levels
  // each row of this table becomes a level tuple, and is the "level" component
  // of some number of elements' coordinates
  // TODO here switch to new perm
  fk::matrix<int> const perm_table =
      opts.use_full_grid
          ? permutations::get_max(pde.num_dims, opts.starting_level, false)
          : permutations::get_lequal(pde.num_dims, opts.starting_level, false);

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
      fk::vector<int> cell_indices =
          index_set.extract_submatrix(cell_set, 0, 1, pde.num_dims);

      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      fk::vector<int> key = level_tuple;
      key.concat(cell_indices);

      // assign into flattened device table builder
      dev_table_builder.concat(key);

      // note the matlab code has an option to append 1d cell indices to the
      // reverse element table. //FIXME do we need to precompute or can we call
      // the 1d helper as needed?
      reverse_table_.push_back(key);
    }
  }

  assert(forward_table_.size() == reverse_table_.size());
  reverse_table_d_.resize(dev_table_builder.size())
      .transfer_from(dev_table_builder);
}

// FIXME need to delete, using index function
// forward lookup - returns the non-negative index of an element's
// coordinates
int element_table::get_index(fk::vector<int> const coords) const
{
  assert(coords.size() > 0);
  // purposely not catching std::out_of_range so that program will die
  return forward_table_.at(coords);
}

// reverse lookup - returns coordinates at a certain index
fk::vector<int> const &element_table::get_coords(int const index) const
{
  assert(index >= 0);
  assert(static_cast<size_t>(index) < reverse_table_.size());
  return reverse_table_[index];
}

// static construction helper
// return the cell indices, given a level tuple
// each row in the returned matrix is the cell portion of an element's
// coordinate
fk::matrix<int>
element_table::get_cell_index_set(fk::vector<int> const &level_tuple)
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

template element_table::element_table(options const program_opts,
                                      PDE<float> const &pde);

template element_table::element_table(options const program_opts,
                                      PDE<double> const &pde);
