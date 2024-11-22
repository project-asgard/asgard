#include "asgard_elements.hpp"

#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "asgard_fast_math.hpp"
#include "asgard_matlab_utilities.hpp"
#include "asgard_program_options.hpp"

namespace asgard::elements
{
// map number of dimensions to max supported level
// FIXME these limits are created by mapping element
// coords to 64 bit integer ids; using a larger
// type for ids in mapping funcs would raise supported
// max levels.

// however, other factors also bound levels, most notably
// kron workspace size and coefficient matrix size...
static std::map<int, int> const dim_to_max_level = {
    {1, 62}, {2, 33}, {3, 20}, {4, 16}, {5, 13}, {6, 9},
};

int64_t get_1d_index(int const level, int const cell)
{
  expect(level >= 0);
  expect(cell >= 0);

  if (level == 0)
  {
    return 0;
  }
  return fm::ipow2(level - 1) + cell;
}

std::array<int64_t, 2> get_level_cell(int64_t const single_dim_id)
{
  expect(single_dim_id >= 0);
  if (single_dim_id == 0)
  {
    return {0, 0};
  }
  auto const level =
      static_cast<int64_t>(std::floor(std::log2(single_dim_id)) + 1);
  auto const cell =
      (level == 0) ? 0 : (single_dim_id - fm::ipow2(level - 1));
  return {level, cell};
}

int64_t map_to_id(fk::vector<int> const &coords, int const max_level,
                  int const num_dims)
{
  expect(max_level > 0);
  expect(num_dims > 0);
  expect(coords.size() == num_dims * 2);
  expect(max_level <= dim_to_max_level.at(num_dims));

  int64_t id     = 0;
  int64_t stride = 1;

  for (auto i = 0; i < num_dims; ++i)
  {
    expect(coords(i) >= 0);
    expect(coords(i) <= max_level);
    expect(coords(i + num_dims) >= 0);

    id += get_1d_index(coords(i), coords(i + num_dims)) * stride;
    stride *= fm::ipow2(max_level);
  }

  expect(id >= 0);
  expect(id <= fm::ipow2(int64_t{max_level} * num_dims));

  return id;
}

fk::vector<int>
map_to_coords(int64_t const id, int const max_level, int const num_dims)
{
  expect(id >= 0);
  expect(max_level > 0);
  expect(num_dims > 0);
  expect(max_level <= dim_to_max_level.at(num_dims));

  auto const stride = fm::ipow2(max_level);

  fk::vector<int> coords(num_dims * 2);
  for (auto i = 0; i < num_dims; ++i)
  {
    auto const id_1d = static_cast<int64_t>(
        std::round((id / static_cast<int64_t>(std::pow(stride, i))) % stride));
    auto const [lev, pos] = get_level_cell(id_1d);
    coords(i)             = lev;
    coords(i + num_dims)  = pos;
  }
  return coords;
}

void table::remove_elements(std::vector<int64_t> const &indices)
{
  if (indices.empty())
  {
    return;
  }

  std::unordered_set<int64_t> const to_delete(indices.begin(), indices.end());

  auto const new_active_ids = [&to_delete,
                               &active_element_ids_ = active_element_ids_]() {
    // don't delete all the elements
    expect(active_element_ids_.size() > to_delete.size());
    std::vector<int64_t> output(active_element_ids_.size() - to_delete.size());
    auto count = 0;
    expect(active_element_ids_.size() < INT_MAX);
    for (auto i = 0; i < static_cast<int>(active_element_ids_.size()); ++i)
    {
      if (to_delete.count(i) == 1)
      {
        continue;
      }
      output[count++] = active_element_ids_[i];
    }
    expect(count == static_cast<int>(output.size()));
    return output;
  }();

  // form new active table from retained elements in old table
  auto const coord_size = static_cast<int64_t>(get_coords(0).size());
  auto const new_table_size =
      active_table_.size() - coord_size * to_delete.size();
  expect(new_table_size > 0);
  auto new_active_table = fk::vector<int>(new_table_size);

  int64_t dest_start = 0;
  expect(size() < INT_MAX);
  for (int64_t i = 0; i < size(); ++i)
  {
    if (to_delete.count(i) == 1)
    {
      continue;
    }
    fk::vector<int, mem_type::const_view> const retained_coords(
        active_table_, i * coord_size, (i + 1) * coord_size - 1);
    auto const dest_stop = dest_start + coord_size - 1;
    fk::vector<int, mem_type::view> dest_for_coords(new_active_table,
                                                    dest_start, dest_stop);
    dest_for_coords = retained_coords;
    dest_start      = dest_stop + 1;
  }

  for (auto const index : indices)
  {
    auto const id = active_element_ids_[index];
    id_to_coords_.erase(id);
  }

  active_table_ = std::move(new_active_table);

  active_element_ids_ = new_active_ids;

  expect(active_element_ids_.size() == id_to_coords_.size());
  expect(size() > 0);
}

int64_t
table::add_elements(std::vector<int64_t> const &ids)
{
  std::unordered_set<int64_t> const child_ids(ids.begin(), ids.end());

  auto active_table_update = active_table_;

  expect(size() > 0);
  auto const coord_size = get_coords(0).size();
  expect(coord_size % 2 == 0);
  auto const num_dims = coord_size / 2;

  int64_t added = 0;
  for (auto const id : ids)
  {
    expect(id >= 0);

    // already present in grid
    if (id_to_coords_.count(id) == 1)
    {
      continue;
    }

    // not present, insert
    auto coords = map_to_coords(id, max_level_, num_dims);
    active_element_ids_.push_back(id);
    // TODO we know a priori how many coords we are adding
    // so this could be optimized away if it's slow
    active_table_update.concat(coords);
    id_to_coords_[id] = std::move(coords);
    added++;
  }
  expect(active_element_ids_.size() == id_to_coords_.size());
  active_table_ = std::move(active_table_update);
  return added;
}

std::list<int64_t>
table::get_child_elements(int64_t const index,
                          std::vector<int> const &max_adapt_level) const
{
  // make sure we're dealing with an active element
  expect(index >= 0);
  expect(index < size());

  auto const coords = get_coords(index);
  // all coordinates have 2 entries (lev, cell) per dimension
  auto const num_dims = coords.size() / 2;

  std::list<int64_t> daughter_ids;
  for (auto i = 0; i < num_dims; ++i)
  {
    // first daughter in this dimension
    int const mlevel = max_adapt_level[i];
    if (coords(i) + 1 <= mlevel)
    {
      auto daughter_coords          = coords;
      daughter_coords(i)            = coords(i) + 1;
      daughter_coords(i + num_dims) = coords(i + num_dims) * 2;
      daughter_ids.push_back(
          map_to_id(daughter_coords, max_level_, num_dims));

      // second daughter
      if (coords(i) >= 1)
      {
        daughter_coords(i + num_dims) = coords(i + num_dims) * 2 + 1;
        daughter_ids.push_back(
            map_to_id(daughter_coords, max_level_, num_dims));
      }
    }
  }
  return daughter_ids;
}

// construct element table
table::table(int const max_level, prog_opts const &options)
    : max_level_(max_level)
{
  expect(max_level_ > 0);
  bool full_grid = (options.grid.value() == grid_type::dense);

  auto const &levels = options.start_levels;
  int const degree   = options.degree.value();
  int const num_dims = static_cast<int>(levels.size());

  auto const perm_table = [&]()
  {
    auto const sort = false;
    if (full_grid)
      return permutations::get_max_multi(levels, num_dims, sort);

    if (options.grid.value() == grid_type::mixed)
    {
      // get maximum level of each group
      fk::vector<int> mixed_max(2);
      mixed_max[0] = *std::max_element(
          std::begin(levels),
          std::next(std::begin(levels), options.mgrid_group.value()));
      mixed_max[1] = *std::max_element(
          std::next(std::begin(levels), options.mgrid_group.value()),
          std::end(levels));

      return permutations::get_mix_leqmax_multi(levels, num_dims, mixed_max,
                                                options.mgrid_group.value(),
                                                sort);
    }

    // default is a simple sparse grid
    return permutations::get_lequal_multi(
        levels, num_dims, *std::max_element(levels.begin(), levels.end()),
        sort);
  }();

  fk::vector<int> dev_table_builder;
  int64_t dof = fm::ipow(degree + 1, num_dims);

  // get a rough DOF estimate used to pre-allocate the element table
  if (full_grid)
  {
    for (auto l : levels)
      dof *= fm::ipow2(l);
  }
  else
  {
    // estimate for sparse grids: deg^ndims * 2^max_lev * max_lev ^ (ndims - 1)
    if (num_dims > 1)
      dof *= fm::ipow2(max_level_) *
             fm::ipow(max_level_, num_dims - 1);
    else
      dof *= fm::ipow2(max_level_);
  }

  // reserve element table data up front
  dev_table_builder.resize(dof);

  int64_t pos = 0;
  for (int row = 0; row < perm_table.nrows(); ++row)
  {
    // get the level tuple to work on
    fk::vector<int> const level_tuple =
        perm_table.extract_submatrix(row, 0, 1, num_dims);
    // calculate all possible cell indices allowed by this level tuple
    fk::matrix<int> const index_set = get_cell_index_set(level_tuple);

    for (int cell_set = 0; cell_set < index_set.nrows(); ++cell_set)
    {
      auto const cell_indices = fk::vector<int>(
          index_set.extract_submatrix(cell_set, 0, 1, num_dims));

      // the element table key is the full element coordinate - (levels,cells)
      // (level-1, ..., level-d, cell-1, ... cell-d)
      auto coords    = fk::vector<int>(level_tuple).concat(cell_indices);
      auto const key = map_to_id(coords, max_level_, num_dims);

      active_element_ids_.push_back(key);

      // assign into flattened device table builder
      if (pos + coords.size() - 1 < dev_table_builder.size())
      {
        dev_table_builder.set_subvector(
            pos, fk::vector<int, mem_type::const_view>(coords));
      }
      else
      {
        // if this is larger than our pre-allocated size, then start resizing
        dev_table_builder.concat(coords);
      }
      pos += coords.size();
      id_to_coords_[key] = std::move(coords);
    }
  }

  if (pos < dev_table_builder.size())
  {
    dev_table_builder = dev_table_builder.extract(0, pos - 1);
  }

  expect(active_element_ids_.size() == id_to_coords_.size());
  active_table_ = std::move(dev_table_builder);
}

void table::recreate_from_elements(std::vector<int64_t> const &element_ids)
{
  // For restarting, we want the element table to contain only the active ids
  // from the restart file. The active ids saved in the restart file is the
  // flattened device table, so we need to recreate the element key from the
  // coordinates.
  std::vector<int64_t> original_ids(active_element_ids_);

  int const coord_size = get_coords(0).size();
  expect(coord_size % 2 == 0);
  int const num_dims = coord_size / 2;

  // calculate the new table size based on the size of each element
  expect(element_ids.size() % coord_size == 0);
  int const new_table_size = static_cast<int>(element_ids.size() / coord_size);

  std::cout << "Recreating element table:\n";
  std::cout << "  - elements from restart: " << new_table_size << "\n";

  // clear the existing hash table
  active_element_ids_.clear();
  id_to_coords_.clear();

  fk::vector<int> dev_table_builder;
  for (int i = 0; i < new_table_size; i++)
  {
    // build a coord set out of the flattened device table
    fk::vector<int> coords(coord_size);
    for (int j = 0; j < coord_size; j++)
    {
      coords(j) = element_ids[j + i * coord_size];
    }

    // get full linear id as key to active element id
    int64_t id = map_to_id(coords, max_level_, num_dims);

    // add the element coords to the flattened device table
    dev_table_builder.concat(coords);

    // add this element to the hash table
    active_element_ids_.push_back(id);
    id_to_coords_[id] = std::move(coords);
  }

  expect(active_element_ids_.size() == id_to_coords_.size());
  active_table_ = std::move(dev_table_builder);

  std::cout << "  - after recreation: " << size() << "\n";
  expect(size() == new_table_size);
}

// static construction helper
// return the cell indices, given a level tuple
// each row in the returned matrix is the cell portion of an element's
// coordinate
fk::matrix<int> table::get_cell_index_set(fk::vector<int> const &level_tuple)
{
  expect(level_tuple.size() > 0);
  for (auto const level : level_tuple)
  {
    ignore(level);
    expect(level >= 0);
  }
  int const num_dims = level_tuple.size();

  // get number of cells for each level coordinate in the input tuple
  // 2^(max(0, level-1))
  fk::vector<int> const cells_per_level = [level_tuple]() {
    fk::vector<int> v(level_tuple.size());
    std::transform(
        level_tuple.begin(), level_tuple.end(), v.begin(),
        [](int level) { return fm::ipow2(std::max(0, level - 1)); });
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

} // namespace asgard::elements
