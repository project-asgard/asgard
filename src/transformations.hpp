#pragma once
#include "batch.hpp"

namespace asgard
{
template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    PDE<P> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
fk::vector<P> gen_realspace_nodes(
    int const degree, int const level, P const min, P const max,
    quadrature_mode const quad_mode = quadrature_mode::use_fixed);

template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    std::vector<dimension<P>> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
void wavelet_to_realspace(
    PDE<P> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
void wavelet_to_realspace(
    std::vector<dimension<P>> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

// overload - get only the elements of the combined vector that fall within a
// specified range
template<typename P>
fk::vector<P>
combine_dimensions(int const, elements::table const &, int const, int const,
                   std::vector<fk::vector<P>> const &, P const = 1.0);

template<typename P>
void combine_dimensions(int const degree, elements::table const &table,
                        int const start_element, int const stop_element,
                        std::vector<fk::vector<P>> const &vectors,
                        P const time_scale,
                        fk::vector<P, mem_type::view> result);

template<typename P, typename F>
fk::vector<P> forward_transform(
    dimension<P> const &dim, F function, g_func_type<P> dv_func,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0)
{
  int const num_levels = dim.get_level();
  int const degree     = dim.get_degree();
  P const domain_min   = dim.domain_min;
  P const domain_max   = dim.domain_max;

  expect(num_levels >= 0);
  expect(num_levels <= transformer.max_level);
  expect(degree >= 0);
  expect(domain_max > domain_min);

  // check to make sure the F function arg is a function type
  // that will accept a vector argument. we have a check for its
  // return below
  static_assert(std::is_invocable_v<decltype(function), fk::vector<P>, P>);

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  auto const [roots, weights] =
      legendre_weights<P>(degree, -1, 1, quadrature_mode::use_fixed);

  // get grid spacing.
  // hate this name TODO
  int const n                  = fm::two_raised_to(num_levels);
  int const degrees_freedom_1d = (degree + 1) * n;

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
      fk::vector<P> out(roots.size());
      std::transform(out.begin(), out.end(), roots.begin(), out.begin(),
                     [&](P &elem, P const &root) {
                       return elem + (normalize * (root / 2.0 + 1.0 / 2.0 + i) +
                                      domain_min);
                     });
      return out;
    }();

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(mapped_roots, time);

    // apply dv to f(v)
    if (dv_func)
    {
      std::transform(f_here.begin(), f_here.end(), mapped_roots.begin(),
                     f_here.begin(),
                     [dv_func, time](P &f_elem, P const &x_elem) -> P {
                       return f_elem * dv_func(x_elem, time);
                     });
    }

    // ensuring function returns vector of appropriate size
    expect(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;

    transformed.set_subvector(i * (degree + 1), coeffs);
  }
  transformed = transformed * (normalize / 2.0);

  // transfer to multi-DG bases
  transformed =
      transformer.apply(transformed, dim.get_level(), basis::side::left,
                        basis::transpose::no_trans);

  // zero out near-zero values resulting from transform to wavelet space
  std::transform(transformed.begin(), transformed.end(), transformed.begin(),
                 [](P &elem) {
                   return std::abs(elem) < std::numeric_limits<P>::epsilon()
                              ? static_cast<P>(0.0)
                              : elem;
                 });

  return transformed;
}

template<typename P>
fk::vector<P> sum_separable_funcs(
    std::vector<md_func_type<P>> const &funcs,
    std::vector<dimension<P>> const &dims,
    adapt::distributed_grid<P> const &grid,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const degree, P const time);

template<typename P>
inline fk::vector<P> transform_and_combine_dimensions(
    PDE<P> const &pde, std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time = 0.0,
    P const time_multiplier = 1.0)
{
  expect(static_cast<int>(v_functions.size()) == pde.num_dims());
  expect(start <= stop);
  expect(stop < table.size());
  expect(degree >= 0);

  std::vector<fk::vector<P>> dimension_components;
  dimension_components.reserve(pde.num_dims());

  auto const &dimensions = pde.get_dimensions();

  for (int i = 0; i < pde.num_dims(); ++i)
  {
    auto const &dim = dimensions[i];
    dimension_components.push_back(forward_transform<P>(
        dim, v_functions[i], dim.volume_jacobian_dV, transformer, time));
    int const n = dimension_components.back().size();
    std::vector<int> ipiv(n);
    expect(dim.get_mass_matrix().nrows() >= n);
    expect(dim.get_mass_matrix().ncols() >= n);
    fk::matrix<P> lhs_mass =
        dim.get_mass_matrix().extract_submatrix(0, 0, n, n);
    fm::gesv(lhs_mass, dimension_components.back(), ipiv);
  }

  return combine_dimensions(degree, table, start, stop, dimension_components,
                            time_multiplier);
}

template<typename P>
inline void transform_and_combine_dimensions(
    std::vector<dimension<P>> const &dims,
    std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time,
    P const time_multiplier, fk::vector<P, mem_type::view> result)

{
  expect(v_functions.size() == static_cast<size_t>(dims.size()) or v_functions.size() == static_cast<size_t>(dims.size() + 1));
  expect(start <= stop);
  expect(stop < table.size());
  expect(degree >= 0);

  std::vector<fk::vector<P>> dimension_components;
  dimension_components.reserve(dims.size());

  for (size_t i = 0; i < dims.size(); ++i)
  {
    auto const &dim = dims[i];
    dimension_components.push_back(forward_transform<P>(
        dim, v_functions[i], dim.volume_jacobian_dV, transformer, time));
    int const n = dimension_components.back().size();
    std::vector<int> ipiv(n);
    fk::matrix<P> lhs_mass = dim.get_mass_matrix();
    expect(lhs_mass.nrows() == n);
    expect(lhs_mass.ncols() == n);
    fm::gesv(lhs_mass, dimension_components.back(), ipiv);
  }

  combine_dimensions(degree, table, start, stop, dimension_components,
                     time_multiplier, result);
}

template<typename P>
inline fk::vector<P> transform_and_combine_dimensions(
    std::vector<dimension<P>> const &dims,
    std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time = 0.0,
    P const time_multiplier = 1.0)
{
  int64_t const vector_size =
      (stop - start + 1) * fm::ipow(degree + 1, dims.size());
  expect(vector_size < INT_MAX);
  fk::vector<P> result(vector_size);
  transform_and_combine_dimensions(dims, v_functions, table, transformer, start,
                                   stop, degree, time, time_multiplier,
                                   fk::vector<P, mem_type::view>(result));
  return result;
}

template<typename P>
inline int dense_space_size(PDE<P> const &pde)
{
  return dense_space_size(pde.get_dimensions());
}

inline int dense_dim_size(int const degree, int const level)
{
  return (degree + 1) * fm::two_raised_to(level);
}

template<typename precision>
inline int dense_space_size(std::vector<dimension<precision>> const &dims)
{
  /* determine the length of the realspace solution */
  int64_t const dense_size = std::accumulate(
      dims.cbegin(), dims.cend(), int64_t{1},
      [](int64_t const size, dimension<precision> const &dim) {
        return size * dense_dim_size(dim.get_degree(), dim.get_level());
      });
  expect(dense_size <= std::numeric_limits<int>::max());
  return static_cast<int>(dense_size);
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
} // namespace asgard
