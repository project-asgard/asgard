#include "boundary_conditions.hpp"
#include <cstdio>

/*

outputs: left_bc_parts and right_bc_parts

the fk::vectors contained in these are generated from the partial terms in the
1D terms that are inside multi-dimensional terms in the PDE.

These two outputs need only be calculated once and can then be used at any time
value "t" to generate the complete boundary condition vector at time "t".

*/
template<typename P>
std::array<unscaled_bc_parts<P>, 2>
boundary_conditions::make_unscaled_bc_parts(PDE<P> const &pde,
                                            element_table const &table,
                                            int const start_element,
                                            int const stop_element,
                                            P const t_init)
{
  assert(start_element >= 0);
  assert(stop_element < table.size());
  assert(stop_element >= start_element);

  unscaled_bc_parts<P> left_bc_parts;
  unscaled_bc_parts<P> right_bc_parts;

  term_set<P> const &terms_vec_vec = pde.get_terms();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  for (int term_num = 0; term_num < static_cast<int>(terms_vec_vec.size());
       ++term_num)
  {
    std::vector<term<P>> const &terms_vec = terms_vec_vec[term_num];

    std::vector<std::vector<fk::vector<P>>> left_dim_pvecs;
    std::vector<std::vector<fk::vector<P>>> right_dim_pvecs;

    for (int dim_num = 0; dim_num < static_cast<int>(dimensions.size());
         ++dim_num)
    {
      dimension<P> const &d = dimensions[dim_num];

      term<P> const &t = terms_vec[dim_num];

      std::vector<fk::vector<P>> left_pvecs;
      std::vector<fk::vector<P>> right_pvecs;

      std::vector<partial_term<P>> const &partial_terms = t.get_partial_terms();
      for (int p_num = 0; p_num < static_cast<int>(partial_terms.size());
           ++p_num)
      {
        partial_term<P> const &p_term = partial_terms[p_num];

        if (p_term.left_homo == homogeneity::inhomogeneous)
        {
          fk::vector<P> trace_bc = compute_left_boundary_condition(
              p_term.g_func, t_init, d, p_term.left_bc_funcs[dim_num]);

          std::vector<fk::vector<P>> p_term_left_bcs = generate_partial_bcs(
              dimensions, dim_num, p_term.left_bc_funcs, t_init, partial_terms,
              p_num, std::move(trace_bc));

          fk::vector<P> combined =
              combine_dimensions(d.get_degree(), table, start_element,
                                 stop_element, p_term_left_bcs);

          left_pvecs.emplace_back(std::move(combined));
        }

        if (p_term.right_homo == homogeneity::inhomogeneous)
        {
          fk::vector<P> trace_bc = compute_right_boundary_condition(
              p_term.g_func, t_init, d, p_term.right_bc_funcs[dim_num]);

          std::vector<fk::vector<P>> p_term_right_bcs = generate_partial_bcs(
              dimensions, dim_num, p_term.right_bc_funcs, t_init, partial_terms,
              p_num, std::move(trace_bc));

          fk::vector<P> combined =
              combine_dimensions(d.get_degree(), table, start_element,
                                 stop_element, p_term_right_bcs);

          right_pvecs.emplace_back(std::move(combined));
        }
      }

      left_dim_pvecs.emplace_back(std::move(left_pvecs));
      right_dim_pvecs.emplace_back(std::move(right_pvecs));
    }

    left_bc_parts.emplace_back(std::move(left_dim_pvecs));
    right_bc_parts.emplace_back(std::move(right_dim_pvecs));
  }

  return {left_bc_parts, right_bc_parts};
}

template<typename P>
fk::vector<P> boundary_conditions::generate_scaled_bc(
    unscaled_bc_parts<P> const &left_bc_parts,
    unscaled_bc_parts<P> const &right_bc_parts, PDE<P> const &pde,
    int const start_element, int const stop_element, P const time)
{
  fk::vector<P> bc(
      (stop_element - start_element + 1) *
      std::pow(pde.get_dimensions()[0].get_degree(), pde.num_dims));

  term_set<P> const &terms_vec_vec = pde.get_terms();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  for (int term_num = 0; term_num < static_cast<int>(terms_vec_vec.size());
       ++term_num)
  {
    std::vector<term<P>> const &terms_vec = terms_vec_vec[term_num];
    for (int dim_num = 0; dim_num < static_cast<int>(dimensions.size());
         ++dim_num)
    {
      term<P> const &t = terms_vec[dim_num];

      std::vector<partial_term<P>> const &partial_terms = t.get_partial_terms();
      int left_index                                    = 0;
      int right_index                                   = 0;
      for (int p_num = 0; p_num < static_cast<int>(partial_terms.size());
           ++p_num)
      {
        partial_term<P> const &p_term = partial_terms[p_num];

        if (p_term.left_homo == homogeneity::inhomogeneous)
        {
          fm::axpy(left_bc_parts[term_num][dim_num][left_index++], bc,
                   p_term.left_bc_time_func(time));
        }

        if (p_term.right_homo == homogeneity::inhomogeneous)
        {
          fm::axpy(right_bc_parts[term_num][dim_num][right_index++], bc,
                   p_term.right_bc_time_func(time));
        }
      }
    }
  }

  return bc;
}

template<typename P>
fk::vector<P> boundary_conditions::compute_left_boundary_condition(
    g_func_type const g_func, P const time, dimension<P> const &dim,
    vector_func<P> const bc_func)
{
  P const domain_min    = dim.domain_min;
  P const domain_max    = dim.domain_max;
  P const domain_extent = domain_max - domain_min;
  assert(domain_extent > 0);

  int const level  = dim.get_level();
  int const degree = dim.get_degree();

  P const total_cells = std::pow(2, level);

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = degree * total_cells;

  fk::vector<P> bc(dof);

  P g = g_func(domain_min, time);
  if (!std::isfinite(g))
  {
    P const small_dx = domain_per_cell * 1e-7;
    g                = g_func(domain_min + small_dx, time);

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    assert(std::isfinite(g));
  }

  /* legendre() returns a 1D matrix - must be converted into a vector */
  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{-1}, degree, legendre_normalization::lin)[0]);

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_min}), time)(0) * g *
                         -1.0;

  legendre_polys_at_value.scale(scale_factor);

  fk::vector<P, mem_type::view> destination_slice(bc, 0, degree - 1);

  assert(destination_slice.size() == legendre_polys_at_value.size());

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc;
}

template<typename P>
fk::vector<P> boundary_conditions::compute_right_boundary_condition(
    g_func_type const g_func, P const time, dimension<P> const &dim,
    vector_func<P> const bc_func)
{
  P const domain_min    = dim.domain_min;
  P const domain_max    = dim.domain_max;
  P const domain_extent = domain_max - domain_min;

  assert(domain_extent > 0);

  int const level  = dim.get_level();
  int const degree = dim.get_degree();

  P const total_cells = std::pow(2, level);

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = degree * total_cells;

  fk::vector<P> bc(dof);

  P g = g_func(domain_max, time);
  if (!std::isfinite(g))
  {
    P const small_dx = domain_per_cell * 1e-7;
    g                = g_func(domain_max - small_dx, time);

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    assert(std::isfinite(g));
  }

  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{1}, degree, legendre_normalization::lin)[0]);

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_max}), time)(0) * g;

  legendre_polys_at_value.scale(scale_factor);

  int const start_index = degree * (total_cells - 1);
  int const stop_index  = degree * total_cells - 1;
  fk::vector<P, mem_type::view> destination_slice(bc, start_index, stop_index);

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc;
}

template<typename P>
std::vector<fk::vector<P>> boundary_conditions::generate_partial_bcs(
    std::vector<dimension<P>> const &dimensions, int const d_index,
    std::vector<vector_func<P>> const &bc_funcs, P const time,
    std::vector<partial_term<P>> const &partial_terms, int const p_index,
    fk::vector<P> &&trace_bc)
{
  assert(d_index < static_cast<int>(dimensions.size()));

  std::vector<fk::vector<P>> partial_bc_vecs;

  for (int dim_num = 0; dim_num < d_index; ++dim_num)
  {
    partial_bc_vecs.emplace_back(
        forward_transform(dimensions[dim_num], bc_funcs[dim_num], time));
  }

  partial_bc_vecs.emplace_back(std::move(trace_bc));

  /* Convert basis operator from double to typename P */
  fm::gemv(fk::matrix<P>(dimensions[d_index].get_to_basis_operator()),
           fk::vector<P>(partial_bc_vecs.back()), partial_bc_vecs.back());

  if (p_index > 0)
  {
    fk::matrix<P> chain = eye<P>(partial_terms[0].get_coefficients().nrows(),
                                 partial_terms[0].get_coefficients().ncols());

    for (int p = 0; p < p_index; ++p)
    {
      fm::gemm(fk::matrix<P>(chain), partial_terms[p].get_coefficients(),
               chain);
    }

    fm::gemv(chain, fk::vector<P>(partial_bc_vecs.back()),
             partial_bc_vecs.back());
  }

  for (int dim_num = d_index + 1; dim_num < static_cast<int>(dimensions.size());
       ++dim_num)
  {
    partial_bc_vecs.emplace_back(
        forward_transform(dimensions[dim_num], bc_funcs[dim_num], time));
  }

  return partial_bc_vecs;
}

/* explicit instantiations */
template std::array<unscaled_bc_parts<double>, 2>
boundary_conditions::make_unscaled_bc_parts(PDE<double> const &pde,
                                            element_table const &table,
                                            int const start_element,
                                            int const stop_element,
                                            double const t_init = 0);

template std::array<unscaled_bc_parts<float>, 2>
boundary_conditions::make_unscaled_bc_parts(PDE<float> const &pde,
                                            element_table const &table,
                                            int const start_element,
                                            int const stop_element,
                                            float const t_init = 0);

template fk::vector<double> boundary_conditions::generate_scaled_bc(
    unscaled_bc_parts<double> const &left_bc_parts,
    unscaled_bc_parts<double> const &right_bc_parts, PDE<double> const &pde,
    int const start_element, int const stop_element, double const time);
template fk::vector<float> boundary_conditions::generate_scaled_bc(
    unscaled_bc_parts<float> const &left_bc_parts,
    unscaled_bc_parts<float> const &right_bc_parts, PDE<float> const &pde,
    int const start_element, int const stop_element, float const time);
template fk::vector<double>
boundary_conditions::compute_left_boundary_condition(
    g_func_type const g_func, double const time, dimension<double> const &dim,
    vector_func<double> const bc_func);
template fk::vector<float> boundary_conditions::compute_left_boundary_condition(
    g_func_type const g_func, float const time, dimension<float> const &dim,
    vector_func<float> const bc_func);

template fk::vector<double>
boundary_conditions::compute_right_boundary_condition(
    g_func_type const g_func, double const time, dimension<double> const &dim,
    vector_func<double> const bc_func);

template fk::vector<float>
boundary_conditions::compute_right_boundary_condition(
    g_func_type const g_func, float const time, dimension<float> const &dim,
    vector_func<float> const bc_func);
template std::vector<fk::vector<double>>
boundary_conditions::generate_partial_bcs(
    std::vector<dimension<double>> const &dimensions, int const d_index,
    std::vector<vector_func<double>> const &bc_funcs, double const time,
    std::vector<partial_term<double>> const &partial_terms, int const p_index,
    fk::vector<double> &&trace_bc);
template std::vector<fk::vector<float>>
boundary_conditions::generate_partial_bcs(
    std::vector<dimension<float>> const &dimensions, int const d_index,
    std::vector<vector_func<float>> const &bc_funcs, float const time,
    std::vector<partial_term<float>> const &partial_terms, int const p_index,
    fk::vector<float> &&trace_bc);
