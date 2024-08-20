#include "asgard_adapt.hpp"
#include "asgard_distribution.hpp"
#include "asgard_transformations.hpp"

namespace asgard::adapt
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

  if (deleted_indices.empty())
  {
    return {};
  }

  std::unordered_set<int64_t> deleted(deleted_indices.begin(),
                                      deleted_indices.end());

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
      new_to_old.insert({new_index - retain_count,
                         grid_limits(old_index - retain_count, old_index - 1)});
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
static void update_levels(elements::table const &adapted_table, PDE<P> &pde)
{
  auto const new_levels =
      get_levels(adapted_table, pde.get_dimensions().size());
  for (auto i = 0; i < static_cast<int>(new_levels.size()); ++i)
  {
    pde.update_dimension(i, new_levels[i]);
  }
}
template<typename P>
static void update_levels(elements::table const &adapted_table,
                          std::vector<dimension<P>> &dims, int const num_terms,
                          std::vector<std::vector<term<P>>> &terms,
                          bool const rechain = false)
{
  auto const new_levels = get_levels(adapted_table, dims.size());
  for (auto i = 0; i < static_cast<int>(new_levels.size()); ++i)
  {
    dims[i].set_level(new_levels[i]);
    if (rechain)
    {
      for (auto j = 0; j < num_terms; ++j)
      {
        terms[j][i].rechain_coefficients(dims[i]);
      }
    }
  }
}

template<typename P>
distributed_grid<P>::distributed_grid(int max_level, prog_opts const &options)
    : table_(max_level, options), max_level_(max_level)
{
  plan_ = get_plan(get_num_ranks(), table_);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen_solution(PDE<P> &pde, fk::vector<P> const &x)
{
  auto const coarse_y = this->coarsen(x, pde.options());
  update_levels(this->get_table(), pde);
  return coarse_y;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine_solution(PDE<P> &pde, fk::vector<P> const &x)
{
  auto const refine_y = this->refine(x, pde.options());
  update_levels(this->get_table(), pde);
  return refine_y;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::redistribute_solution(fk::vector<P> const &x,
                                           distribution_plan const &old_plan,
                                           int const old_size)
{
  return redistribute_vector(x, old_plan, plan_, remap_for_addtl(old_size));
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine(fk::vector<P> const &x, prog_opts const &options)
{
  adapt_norm const anorm = options.anorm.value();

  P const max_elem   = anorm == adapt_norm::linf ? fm::nrminf(x) : fm::nrm2(x);
  P const global_max = get_global_max<P>(max_elem, this->plan_);

  auto const refine_threshold = options.adapt_threshold.value() * global_max;
  if (refine_threshold <= 0.0)
  {
    return x;
  }

  auto const refine_check =
      [anorm, refine_threshold](
          int64_t const, fk::vector<P, mem_type::const_view> const &element_x) {
        auto const refined_max_elem =
            anorm == adapt_norm::linf ? fm::nrminf(element_x) : fm::nrm2(element_x);
        return refined_max_elem >= refine_threshold;
      };
  auto const to_refine = filter_elements(refine_check, x);
  return this->refine_elements(to_refine, options.max_levels, x);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen(fk::vector<P> const &x, prog_opts const &options)
{
  adapt_norm const anorm = options.anorm.value();

  P const max_elem   = anorm == adapt_norm::linf ? fm::nrminf(x) : fm::nrm2(x);
  P const global_max = get_global_max<P>(max_elem, this->plan_);
  P const refine_threshold = options.adapt_threshold.value() * global_max;
  if (refine_threshold <= 0.0)
  {
    return x;
  }

  auto const coarsen_threshold = refine_threshold * 0.1;
  auto const &table            = this->table_;
  auto const coarsen_check =
      [&table, anorm, coarsen_threshold](
          int64_t const elem_index,
          fk::vector<P, mem_type::const_view> const &element_x) {
        P const coarsened_max_elem =
            anorm == adapt_norm::linf ? fm::nrminf(element_x) : fm::nrm2(element_x);
        auto const coords    = table.get_coords(elem_index);
        auto const min_level = *std::min_element(
            coords.begin(), coords.begin() + coords.size() / 2);
        return std::abs(coarsened_max_elem) <= coarsen_threshold &&
               min_level >= 0;
      };

  auto const to_coarsen = filter_elements(coarsen_check, x);
  return this->remove_elements(to_coarsen, x);
}

template<typename P>
fk::vector<P> distributed_grid<P>::refine_elements(
    std::vector<int64_t> const &indices_to_refine,
    std::vector<int> const &max_levels, fk::vector<P> const &x)
{
  std::list<int64_t> child_ids;
  for (auto const parent_index : indices_to_refine)
  {
    child_ids.splice(child_ids.end(),
                     table_.get_child_elements(parent_index, max_levels));
  }

  // need to preserve ordering for testing
  auto const get_unique = [](auto const &ids) {
    std::unordered_set<int64_t> ids_so_far;
    std::vector<int64_t> unique_ids;
    for (auto const id : ids)
    {
      if (ids_so_far.count(id) == 0)
      {
        unique_ids.push_back(id);
      }
      ids_so_far.insert(id);
    }
    return unique_ids;
  };

  auto const all_child_ids =
      get_unique(distribute_table_changes(get_unique(child_ids), plan_));

  if (all_child_ids.empty())
  {
    return x;
  }

  auto const added    = table_.add_elements(all_child_ids);
  auto new_plan       = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_addtl(table_.size() - added);
  fk::vector<P> y     = redistribute_vector(x, plan_, new_plan, remapper);

  plan_ = std::move(new_plan);

  return y;
}

template<typename P>
fk::vector<P> distributed_grid<P>::remove_elements(
    std::vector<int64_t> const &indices_to_remove, fk::vector<P> const &x)
{
  auto const all_remove_indices =
      distribute_table_changes(indices_to_remove, plan_);

  if (all_remove_indices.empty())
  {
    return x;
  }

  table_.remove_elements(all_remove_indices);
  if (this->size() < get_num_ranks())
  {
    node_out() << "coarsened below number of ranks - can't handle this case yet"
               << '\n';
  }
  auto new_plan       = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_delete(all_remove_indices, table_.size());
  fk::vector<P> y     = redistribute_vector(x, plan_, new_plan, remapper);

  plan_ = std::move(new_plan);

  return y;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class distributed_grid<double>;
#endif
#ifdef ASGARD_ENABLE_FLOAT
template class distributed_grid<float>;
#endif

} // namespace asgard::adapt
