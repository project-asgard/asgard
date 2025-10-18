#include "asgard_term_manager.hpp"

#include "asgard_blas.hpp"

namespace asgard
{

template<typename P>
term_entry<P>::term_entry(term_md<P> tin)
  : tmd(std::move(tin)), has_poisson(false)
{
  expect(not tmd.is_chain());
  if (tmd.is_interpolatory()) {
    return; // interpolation poisson dependence goes here
  }

  int const num_dims = tmd.num_dims();
  std::vector<int> active_dirs;
  active_dirs.reserve(num_dims);
  int flux_dir = -1;
  for (int d : iindexof(num_dims))
  {
    auto const &t1d = tmd.dim(d);
    if (not t1d.is_identity()) {
      active_dirs.push_back(d);
      if (t1d.has_flux()) {
        flux_dir = d;
        if (active_dirs.size() > 1)
          std::swap(active_dirs.front(), active_dirs.back());
      }
    }

    has_poisson = has_poisson or has_needs_poisson(t1d);
  }

  perm = kronmult::permutes(active_dirs, flux_dir);
}

template<typename P>
bool term_entry<P>::has_needs_poisson(term_1d<P> const &t1d) {
  auto check_poisson = [](term_1d<P> const &single)
    -> bool {
      return (single.depends() == term_dependence::electric_field or
              single.depends() == term_dependence::electric_field_only);
    };

  if (t1d.is_chain()) {
    for (int i : iindexof(t1d.num_chain()))
      if (check_poisson(t1d[i]))
        return true;
    return false;
  } else {
    return check_poisson(t1d);
  }
}


}
