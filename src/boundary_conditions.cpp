#include "boundary_conditions.hpp"
#include <cstdio>

template<typename P>
bc_timestepper<P>::bc_timestepper(PDE<P> const &pde, element_table const &table,
                                  int const start_element,
                                  int const stop_element, P const t_init)
    : t_init(t_init),
      bc_size((stop_element - start_element + 1) *
              std::pow(pde.get_dimensions()[0].get_degree(), pde.num_dims)),
      pde(pde)
{
  assert(bc_size > 0);

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
              p_term.g_func, t_init, d.get_level(), d.get_degree(),
              d.domain_min, d.domain_max, p_term.left_bc_funcs[dim_num]);

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
              p_term.g_func, t_init, d.get_level(), d.get_degree(),
              d.domain_min, d.domain_max, p_term.right_bc_funcs[dim_num]);

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

    left.emplace_back(std::move(left_dim_pvecs));
    right.emplace_back(std::move(right_dim_pvecs));
  }

  return;
}

template<typename P>
fk::vector<P> bc_timestepper<P>::advance(P const time) const
{
  fk::vector<P> bc(bc_size);

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
          fm::axpy(left[term_num][dim_num][left_index++], bc,
                   p_term.left_bc_time_func(time));
        }

        if (p_term.right_homo == homogeneity::inhomogeneous)
        {
          fm::axpy(right[term_num][dim_num][right_index++], bc,
                   p_term.right_bc_time_func(time));
        }
      }
    }
  }

  return bc;
}

template<typename P>
std::vector<fk::vector<P>> bc_timestepper<P>::generate_partial_bcs(
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

template<typename P>
fk::vector<P> bc_timestepper<P>::compute_right_boundary_condition(
    g_func_type const g_func, P const time, int level, int degree,
    P const domain_min, P const domain_max, vector_func<P> const bc_func)
{
  P const domain_extent = domain_max - domain_min;
  assert(domain_extent > 0);

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

  /* Use the new upcoming vector-view of matrix column constructor to
   * avoid this copy */
  /* Currently in Tyler's gmres PR - not yet merged into develop */
  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{1}, degree, legendre_normalization::lin)[0]);

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_max}), time)(0) * g;

  legendre_polys_at_value.scale(scale_factor);

  int const start_index = degree * (total_cells - 1);
  int const stop_index  = degree * total_cells - 1;
  fk::vector<P, mem_type::view> destination_slice(bc, start_index, stop_index);

  assert(destination_slice.size() == legendre_polys_at_value.size());

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc;
}

template<typename P>
fk::vector<P> bc_timestepper<P>::compute_left_boundary_condition(
    g_func_type const g_func, P const time, int level, int degree,
    P const domain_min, P const domain_max, vector_func<P> const bc_func)
{
  P const domain_extent = domain_max - domain_min;
  assert(domain_extent > 0);

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

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_min}), time)(0) * g *
                         -1.0;

  /* legendre() returns a 1D matrix - must be converted into a vector */
  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{-1}, degree, legendre_normalization::lin)[0]);

  legendre_polys_at_value.scale(scale_factor);

  fk::vector<P, mem_type::view> destination_slice(bc, 0, degree - 1);

  assert(destination_slice.size() == legendre_polys_at_value.size());

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc;
}

/* explicit instantiations */
template class bc_timestepper<float>;
template class bc_timestepper<double>;
