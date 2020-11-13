#include "adapt.hpp"
#include "distribution.hpp"

#include <unordered_set>

namespace adapt
{
static std::map<int64_t, grid_limits>
remap_for_addtl(int64_t const old_num_elems)
{
  assert(old_num_elems > 0);
  std::map<int64_t, grid_limits> mapper;
  // beginning of new elem range maps directly to old elem range
  mapper.insert({0, grid_limits(0, old_num_elems - 1)});
  return mapper;
}

static std::map<int64_t, grid_limits>
remap_for_delete(std::vector<int64_t> const &deleted_indices,
                 int64_t const num_new_elems)
{
  assert(num_new_elems > 0);

  if (deleted_indices.size() == 0)
  {
    return {};
  }

  std::unordered_set<int64_t> deleted(deleted_indices.begin(),
                                      deleted_indices.end());
  assert(deleted.size() ==
         deleted_indices.size()); // should all be unique indices
  std::map<int64_t, grid_limits> new_to_old;
  int64_t old_index    = 0;
  int64_t new_index    = 0;
  int64_t retain_count = 0;

  while (new_index < num_new_elems)
  {
    // while in a preserved region, advance both indices
    // count how many elements are in the region
    while (deleted.count(old_index) == 0)
    {
      old_index++;
      new_index++;
      retain_count++;
      if (new_index >= num_new_elems - 1)
      {
        break;
      }
    }

    // record preserved region
    if (retain_count > 0)
    {
      new_to_old.insert({new_index - retain_count - 1,
                         grid_limits(old_index - retain_count - 1, old_index)});
      retain_count = 0;
    }

    // skip past deleted regions
    while (deleted.count(old_index) == 1)
    {
      old_index++;
    }
  }

  return new_to_old;
}

template<typename P>
distributed_grid<P>::distributed_grid(options const &cli_opts,
                                      PDE<P> const &pde)
    : table_(cli_opts, pde)
{
  plan_ = get_plan(get_num_ranks(), table_);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine(fk::vector<P> const &x, options const &cli_opts)
{
  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const max_elem = *std::max_element(x.begin(), x.end(), abs_compare);
  auto const refine_threshold = cli_opts.adapt_threshold * max_elem;
  if (refine_threshold <= 0.0)
  {
    return x;
  }

  auto const refine_check =
      [refine_threshold, abs_compare](
          int64_t const, fk::vector<P, mem_type::const_view> const &element_x) {
        auto const max_elem =
            *std::max_element(element_x.begin(), element_x.end(), abs_compare);
        return max_elem >= refine_threshold;
      };

  auto const to_refine = filter_elements(refine_check, x);
  return this->refine_elements(to_refine, cli_opts, x);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen(fk::vector<P> const &x, options const &cli_opts)
{
  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const max_elem = *std::max_element(x.begin(), x.end(), abs_compare);

  auto const refine_threshold = cli_opts.adapt_threshold * max_elem;
  if (refine_threshold <= 0.0)
  {
    return x;
  }
  auto const coarsen_threshold = refine_threshold * 0.1;

  auto const &table = this->table_;
  auto const coarsen_check =
      [&table, coarsen_threshold,
       abs_compare](int64_t const elem_index,
                    fk::vector<P, mem_type::const_view> const &element_x) {
        auto const max_elem =
            *std::max_element(element_x.begin(), element_x.end(), abs_compare);
        auto const coords    = table.get_coords(elem_index);
        auto const min_level = *std::min_element(coords.begin(), coords.end());
        return max_elem <= coarsen_threshold && min_level >= 2;
      };

  auto const to_coarsen = filter_elements(coarsen_check, x);
  return this->remove_elements(to_coarsen, x);
}

template<typename P>
fk::vector<P> distributed_grid<P>::refine_elements(
    std::vector<int64_t> const &indices_to_refine, options const &opts,
    fk::vector<P> const &x)
{
  std::vector<int64_t> child_ids;
  for (auto const parent_index : indices_to_refine)
  {
    auto const children = table_.get_child_elements(parent_index, opts);
    child_ids.insert(child_ids.end(), children.begin(), children.end());
  }

  auto const all_child_ids = distribute_table_changes(child_ids, plan_);

  if (all_child_ids.size() == 0)
  {
    return x;
  }

  table_.add_elements(all_child_ids, opts.max_level);

  auto const new_plan = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_addtl(table_.size() - all_child_ids.size());
  auto const y        = redistribute_vector(x, plan_, new_plan, remapper);
  plan_               = new_plan;

  return y;
}

template<typename P>
fk::vector<P> distributed_grid<P>::remove_elements(
    std::vector<int64_t> const &indices_to_remove, fk::vector<P> const &x)
{
  auto const all_remove_indices =
      distribute_table_changes(indices_to_remove, plan_);

  if (all_remove_indices.size() == 0)
  {
    return x;
  }

  table_.remove_elements(all_remove_indices);

  auto const new_plan = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_delete(all_remove_indices, table_.size());
  auto const y        = redistribute_vector(x, plan_, new_plan, remapper);
  plan_               = new_plan;

  // TODO rechain pde...
  return y;
}

template class distributed_grid<float>;
template class distributed_grid<double>;

} // namespace adapt
