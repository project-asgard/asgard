#pragma once

#include "basis.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "fast_math.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

template<typename P>
fk::matrix<P>
recursive_kron(std::vector<fk::matrix<P, mem_type::view>> &kron_matrices,
               int const index = 0);

template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    PDE<P> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer);

template<typename P>
void wavelet_to_realspace(
    PDE<P> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const memory_limit_MB,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space);

template<typename P>
fk::vector<P>
combine_dimensions(int const, elements::table const &element_table,
                   std::vector<fk::vector<P>> const &, P const = 1.0);

// overload - get only the elements of the combined vector that fall within a
// specified range
template<typename P>
fk::vector<P>
combine_dimensions(int const, elements::table const &, int const, int const,
                   std::vector<fk::vector<P>> const &, P const = 1.0);

template<typename P, typename F>
fk::vector<P> forward_transform(
    dimension<P> const &dim, F function, g_func_type<P> const dv_func,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const t = 0)
{
  int const num_levels = dim.get_level();
  int const degree     = dim.get_degree();
  P const domain_min   = dim.domain_min;
  P const domain_max   = dim.domain_max;

  expect(num_levels >= 0);
  expect(num_levels <= transformer.max_level);
  expect(degree > 0);
  expect(domain_max > domain_min);

  // check to make sure the F function arg is a function type
  // that will accept a vector argument. we have a check for its
  // return below
  static_assert(std::is_invocable_v<decltype(function), fk::vector<P>, P>);

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  auto const [roots, weights] = legendre_weights<P>(degree, -1, 1);

  // get grid spacing.
  // hate this name TODO
  int const n                  = fm::two_raised_to(num_levels);
  int const degrees_freedom_1d = degree * n;

  // get the Legendre basis function evaluated at the Legendre-Gauss nodes   //
  // up to order k
  P const normalize         = (domain_max - domain_min) / n;
  fk::matrix<P> const basis = [&roots = roots, degree, normalize] {
    fk::matrix<P> legendre_ = legendre<P>(roots, degree)[0];
    return legendre_.transpose() * (static_cast<P>(1.0) / std::sqrt(normalize));
  }();

  // this will be our return vector
  fk::vector<P> transformed(degrees_freedom_1d);

  // initial condition for f
  // hate this name also TODO

  for (int i = 0; i < n; ++i)
  {
    // map quad_x from [-1,+1] to [domain_min,domain_max] physical domain.
    fk::vector<P> const mapped_roots = [&roots = roots, normalize, domain_min,
                                        i]() {
      fk::vector<P> mapped_roots(roots.size());
      std::transform(mapped_roots.begin(), mapped_roots.end(), roots.begin(),
                     mapped_roots.begin(), [&](P &elem, P const &root) {
                       return elem + (normalize * (root / 2.0 + 1.0 / 2.0 + i) +
                                      domain_min);
                     });
      return mapped_roots;
    }();

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(mapped_roots, t);

    // apply dv to f(v)
    std::transform(f_here.begin(), f_here.end(), mapped_roots.begin(),
                   f_here.begin(),
                   [dv_func, t](P &f_elem, P const &x_elem) -> P {
                     return f_elem * dv_func(x_elem, t);
                   });

    // ensuring function returns vector of appropriate size
    expect(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;

    transformed.set_subvector(i * degree, coeffs);
  }
  transformed = transformed * (normalize / 2.0);

  // transfer to multi-DG bases
  transformed =
      transformer.apply(transformed, dim.get_level(), basis::side::left,
                        basis::transpose::no_trans);

  // zero out near-zero values resulting from transform to wavelet space
  std::transform(transformed.begin(), transformed.end(), transformed.begin(),
                 [](P &elem) {
                   P const compare = [] {
                     if constexpr (std::is_same<P, double>::value)
                     {
                       return static_cast<P>(1e-12);
                     }
                     return static_cast<P>(1e-4);
                   }();
                   return std::abs(elem) < compare ? static_cast<P>(0.0) : elem;
                 });

  return transformed;
}

template<typename P>
inline fk::vector<P> transform_and_combine_dimensions(
    PDE<P> const &pde, std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time = 0.0,
    P const time_multiplier = 1.0)
{
  expect(static_cast<int>(v_functions.size()) == pde.num_dims);
  expect(start <= stop);
  expect(stop < table.size());
  expect(degree > 0);

  std::vector<fk::vector<P>> dimension_components;
  dimension_components.reserve(pde.num_dims);

  auto const &dimensions = pde.get_dimensions();

  for (int i = 0; i < pde.num_dims; ++i)
  {
    auto const &dim = dimensions[i];
    dimension_components.push_back(forward_transform<P>(
        dim, v_functions[i], dim.moment_dV, transformer, time));
    int const n = dimension_components.back().size();
    std::vector<int> ipiv(n);
    fk::matrix<P, mem_type::const_view> const lhs_mass(dim.get_mass_matrix(), 0,
                                                       n - 1, 0, n - 1);
    expect(lhs_mass.nrows() == n);
    expect(lhs_mass.ncols() == n);
    fm::gesv(lhs_mass, dimension_components.back(), ipiv);
  }

  return combine_dimensions(degree, table, start, stop, dimension_components,
                            time_multiplier);
}

template<typename P>
inline int real_solution_size(PDE<P> const &pde)
{
  /* determine the length of the realspace solution */
  std::vector<dimension<P>> const &dims = pde.get_dimensions();
  int prod                              = 1;
  for (int i = 0; i < static_cast<int>(dims.size()); i++)
  {
    prod *= (dims[i].get_degree() * std::pow(2, dims[i].get_level()));
  }

  return prod;
}

template<typename P>
static std::array<fk::vector<P, mem_type::view, resource::host>, 2>
update_transform_workspace(
    int const sol_size,
    fk::vector<P, mem_type::owner, resource::host> &workspace,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> const
        &transform_wksp)
{
  if (transform_wksp[0].size() < sol_size)
  {
    workspace.resize(sol_size * 2);
    return std::array<fk::vector<P, mem_type::view, resource::host>, 2>{
        fk::vector<P, mem_type::view, resource::host>(workspace, 0,
                                                      sol_size - 1),
        fk::vector<P, mem_type::view, resource::host>(workspace, sol_size,
                                                      sol_size * 2 - 1)};
  }
  expect(workspace.size() >= sol_size * 2);
  expect(transform_wksp[0].size() >= sol_size);
  expect(transform_wksp[1].size() >= sol_size);
  return transform_wksp;
}
