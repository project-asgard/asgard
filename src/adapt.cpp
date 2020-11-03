#include "adapt.hpp"
#include "distribution.hpp"

namespace adapt
{
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
      [refine_threshold,
       abs_compare](int64_t const,
                    fk::vector<P, mem_type::const_view> const &element_x) {
         auto const max_elem =
            *std::max_element(element_x.begin(), element_x.end(), abs_compare);
        return max_elem >= refine_threshold;
      };

  auto const to_refine = filter_elements(refine_check, x);
  return this->refine_elements(to_refine, x);
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

  auto const& table = this->table_;
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
fk::vector<P>
distributed_grid<P>::refine_elements(std::vector<int64_t> indices_to_refine, fk::vector<P> const &x)
{
  return x;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::remove_elements(std::vector<int64_t> indices_to_remove, fk::vector<P> const &x)
{
  return x;
}

template class distributed_grid<float>;
template class distributed_grid<double>;

} // namespace adapt
