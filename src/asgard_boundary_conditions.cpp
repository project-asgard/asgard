#include "asgard_boundary_conditions.hpp"

#include "asgard_small_mats.hpp"

/*

outputs: left_bc_parts and right_bc_parts

the fk::vectors contained in these are generated from the partial terms in the
1D terms that are inside multi-dimensional terms in the PDE.

These two outputs need only be calculated once and can then be used at any time
value "t" to generate the complete boundary condition vector at time "t".

*/
namespace asgard::boundary_conditions
{
template<typename P>
std::vector<std::vector<P>> generate_partial_bcs(
    std::vector<dimension<P>> const &dimensions, int const d_index,
    std::vector<vector_func<P>> const &bc_funcs,
    hierarchy_manipulator<P> const &hier, coefficient_matrices<P> &cmats,
    connection_patterns const &conn,
    P const time, std::vector<term<P>> const &term_md,
    std::vector<partial_term<P>> const &,
    int const t_index, int const p_index, std::vector<P> &&trace_bc)
{
  expect(d_index < static_cast<int>(dimensions.size()));

  int const num_dims = static_cast<int>(dimensions.size());
  int const degree   = dimensions.front().get_degree();

  std::vector<std::vector<P>> partial_bc_vecs;

  for (int d : indexof<int>(num_dims))
  {
    int const level = dimensions[d].get_level();
    auto const &f = bc_funcs[d];
    auto const &g = term_md[d].get_partial_terms()[p_index].g_func();
    vector_func<P> bc_func;
    if (g)
    {
      bc_func = [f, g](fk::vector<P> const &x, P const &t) {
        // evaluate f(x,t) * g(x,t)
        fk::vector<P> fx(f(x, t));
        std::transform(fx.begin(), fx.end(), x.begin(), fx.begin(),
                       [g, t](P &f_elem, P const &x_elem) -> P {
                         return f_elem * g(x_elem, t);
                       });
        return fx;
      };
    }
    else
    {
      bc_func = f;
    }

    function_1d<P> dv = nullptr;
    if (term_md[d].get_partial_terms()[p_index].dv_func())
      dv = [&](std::vector<P> const &x, std::vector<P> &fx)
                -> void {
              for (size_t i = 0; i < x.size(); i++)
                fx[i] = term_md[d].get_partial_terms()[p_index].dv_func()(x[i], time);
            };

    // this applies the mass matrix too
    hier.project1d_f(
      [&](std::vector<P> const &x, std::vector<P> &fx)
          -> void {
        auto fkvec = bc_func(x, time);
        std::copy(fkvec.begin(), fkvec.end(), fx.data());
      }, dv, cmats.pterm_mass[t_index * num_dims + d][p_index], d, level);
    partial_bc_vecs.push_back(hier.get_projected1d(d));

    // Apply previous pterms
    for (int p_num = 0; p_num < p_index; ++p_num)
    {
      fk::vector<P> const bcopy = partial_bc_vecs.back();
      cmats.pterm_coeffs[t_index * num_dims + d][p_num].gemv(degree + 1, level, conn, bcopy.data(),
                                                             partial_bc_vecs.back().data());
    }
  }

  int const level = dimensions[d_index].get_level();
  partial_bc_vecs[d_index] = std::move(trace_bc);

  hier.project1d(level, partial_bc_vecs[d_index]);

  if (cmats.pterm_mass[t_index * num_dims + d_index][p_index].has_level(level))
  {
    invert_mass(degree + 1, cmats.pterm_mass[t_index * num_dims + d_index][p_index][level],
                partial_bc_vecs[d_index].data());
  }

  for (int p = 0; p < p_index; ++p)
  {
    fk::vector<P> const bcopy = partial_bc_vecs[d_index];
    cmats.pterm_coeffs[t_index * num_dims + d_index][p].gemv(
        degree + 1, level, conn, bcopy.data(), partial_bc_vecs[d_index].data());
  }

  return partial_bc_vecs;
}

// FIXME refactor this component. the whole thing.
template<typename P>
std::array<unscaled_bc_parts<P>, 2> make_unscaled_bc_parts(
    PDE<P> const &pde, elements::table const &table,
    hierarchy_manipulator<P> const &hier, coefficient_matrices<P> &cmats,
    connection_patterns const &conn,
    int const start_element, int const stop_element, P const t_init)
{
  tools::time_event timing("make unscaled bc");
  expect(start_element >= 0);
  expect(stop_element < table.size());
  expect(stop_element >= start_element);

  unscaled_bc_parts<P> left_bc_parts;
  unscaled_bc_parts<P> right_bc_parts;

  term_set<P> const &terms_vec_vec = pde.get_terms();

  int const num_terms = static_cast<int>(terms_vec_vec.size());
  int const num_dims  = pde.num_dims();

  std::vector<dimension<P>> const &dimensions = pde.get_dimensions();

  int const degree = dimensions.front().get_degree();

  int const vec_size = (stop_element - start_element + 1) * fm::ipow(degree + 1, num_dims);

  for (int t : indexof<int>(num_terms))
  {
    std::vector<term<P>> const &term_md = terms_vec_vec[t];

    std::vector<std::vector<std::vector<P>>> left_dim_pvecs;
    std::vector<std::vector<std::vector<P>>> right_dim_pvecs;

    for (int d : indexof<int>(num_dims))
    {
      dimension<P> const &dim = dimensions[d];

      term<P> const &term = term_md[d];

      std::vector<std::vector<P>> left_pvecs;
      std::vector<std::vector<P>> right_pvecs;

      std::vector<partial_term<P>> const &pterms = term.get_partial_terms();
      for (int pt : indexof<int>(pterms.size()))
      {
        partial_term<P> const &pterm = pterms[pt];

        if (not pterm.left_bc_zero())
        {
          std::vector<P> trace_bc = compute_left_boundary_condition(
              pterm.g_func(), pterm.dv_func(), t_init, dim,
              pterm.left_bc_funcs()[d]);

          std::vector<std::vector<P>> p_term_left_bcs = generate_partial_bcs(
              dimensions, d, pterm.left_bc_funcs(), hier, cmats,
              conn, t_init, term_md, pterms, t, pt, std::move(trace_bc));

          std::vector<P> combined(vec_size);

          combine_dimensions(degree, table, start_element,
                             stop_element, p_term_left_bcs, combined.data());

          left_pvecs.emplace_back(std::move(combined));
        }

        if (not pterm.right_bc_zero())
        {
          std::vector<P> trace_bc = compute_right_boundary_condition(
              pterm.g_func(), pterm.dv_func(), t_init, dim,
              pterm.right_bc_funcs()[d]);

          std::vector<std::vector<P>> p_term_right_bcs = generate_partial_bcs(
              dimensions, d, pterm.right_bc_funcs(), hier, cmats,
              conn, t_init, term_md, pterms, t, pt, std::move(trace_bc));

          std::vector<P> combined(vec_size);

          combine_dimensions(degree, table, start_element,
                             stop_element, p_term_right_bcs, combined.data());

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

// template<typename P>
// bc_manager<P>::bc_manager(PDE<P> const &pde, elements::table const &table,
//     hierarchy_manipulator<P> const &hier, coefficient_matrices<P> &cmats,
//     connection_patterns const &conns,
//     int const start_element, int const stop_element, P const t_init)
//   : num_dims(pde.num_dims()), degree(hier.degree())
// {
//   tools::time_event timing("make unscaled bc");
//   expect(start_element >= 0);
//   expect(stop_element < table.size());
//   expect(stop_element >= start_element);
//
//   // count the number of boundary condition vectors that we will need
//   // the first vector (if present) is left unscaled, i.e., assumed 1
//   // the others will use corresponding functions from the p-terms
//   // so, we first count the number of basis that we will need
//   int num_pt    = 0;
//
//   for (auto const &term : pde.get_terms()) {
//     for (int d : indexof<int>(pde.num_dims())) {
//       for (auto const &pt : term[d].get_partial_terms()) {
//         if (not pt.left_bc_zero())
//         {
//           if (pt.left_bc_time_func())
//             num_pt++;
//           else
//             num_const = 1;
//         }
//         if (not pt.right_bc_zero())
//         {
//           if (pt.right_bc_time_func())
//             num_pt++;
//           else
//             num_const = 1;
//         }
//       }
//     }
//   }
//
//   std::vector<dimension<P>> const &dimensions = pde.get_dimensions();
//   int const degree = hier.degree();
//
//   time_funcs.reserve(num_pt);
//
//   int64_t const vec_size = (stop_element - start_element + 1) * hier.block_size();
//   unscaled_bc.resize(vec_size * (num_pt + num_const));
//
//   std::vector<P> temp;
//
//   P *unc = unscaled_bc.data();
//   if (num_const > 0) {
//     unc += vec_size;
//     temp.resize(vec_size);
//   }
//
//   for (int t : indexof<int>(pde.num_terms()))
//   {
//     std::vector<term<P>> const &term_md = pde.get_terms()[t];
//
//     for (int d : indexof<int>(pde.num_dims()))
//     {
//       dimension<P> const &dim = dimensions[d];
//       term<P> const &term     = term_md[d];
//
//       std::vector<partial_term<P>> const &pterms = term.get_partial_terms();
//       for (int pt : indexof<int>(pterms.size()))
//       {
//         auto const &ptt = pterms[pt];
//
//         if (not ptt.left_bc_zero())
//         {
//           std::vector<P> trace_bc = compute_left_boundary_condition(
//               ptt.g_func(), ptt.dv_func(), t_init, dim,
//               ptt.left_bc_funcs()[d]);
//
//           std::vector<std::vector<P>> p_term_left_bcs = generate_partial_bcs(
//               dimensions, d, ptt.left_bc_funcs(), hier, cmats,
//               conns, t_init, term_md, term.get_partial_terms(), t, pt, std::move(trace_bc));
//
//           if (ptt.left_bc_time_func())
//             num_pt++;
//           else
//             num_const = 1;
//         }
//         if (not ptt.right_bc_zero())
//         {
//           if (ptt.right_bc_time_func())
//             num_pt++;
//           else
//             num_const = 1;
//         }
//       }
//     }
//   }
//
//
//
//   unscaled_bc_parts<P> left_bc_parts;
//   unscaled_bc_parts<P> right_bc_parts;
//
//   term_set<P> const &terms_vec_vec = pde.get_terms();
//
//   int const num_terms = static_cast<int>(terms_vec_vec.size());
//   int const num_dims  = pde.num_dims();
//
//
//   //int const degree = dimensions.front().get_degree();
//
//   for (int t : indexof<int>(num_terms))
//   {
//     std::vector<term<P>> const &term_md = terms_vec_vec[t];
//
//     std::vector<std::vector<std::vector<P>>> left_dim_pvecs;
//     std::vector<std::vector<std::vector<P>>> right_dim_pvecs;
//
//     for (int d : indexof<int>(num_dims))
//     {
//       dimension<P> const &dim = dimensions[d];
//
//       term<P> const &term = term_md[d];
//
//       std::vector<std::vector<P>> left_pvecs;
//       std::vector<std::vector<P>> right_pvecs;
//
//       std::vector<partial_term<P>> const &pterms = term.get_partial_terms();
//       for (int pt : indexof<int>(pterms.size()))
//       {
//         partial_term<P> const &pterm = pterms[pt];
//
//         if (not pterm.left_bc_zero())
//         {
//           std::vector<P> trace_bc = compute_left_boundary_condition(
//               pterm.g_func(), pterm.dv_func(), t_init, dim,
//               pterm.left_bc_funcs()[d]);
//
//           std::vector<std::vector<P>> p_term_left_bcs = generate_partial_bcs(
//               dimensions, d, pterm.left_bc_funcs(), hier, cmats,
//               conns, t_init, term_md, pterms, t, pt, std::move(trace_bc));
//
//           // std::vector<P> combined =
//           //     combine_dimensions(degree, table, start_element,
//           //                        stop_element, p_term_left_bcs);
//           //
//           // left_pvecs.emplace_back(std::move(combined));
//         }
//
//         if (not pterm.right_bc_zero())
//         {
//           std::vector<P> trace_bc = compute_right_boundary_condition(
//               pterm.g_func(), pterm.dv_func(), t_init, dim,
//               pterm.right_bc_funcs()[d]);
//
//           std::vector<std::vector<P>> p_term_right_bcs = generate_partial_bcs(
//               dimensions, d, pterm.right_bc_funcs(), hier, cmats,
//               conns, t_init, term_md, pterms, t, pt, std::move(trace_bc));
//
//           // std::vector<P> combined =
//           //     combine_dimensions(degree, table, start_element,
//           //                        stop_element, p_term_right_bcs);
//           //
//           // right_pvecs.emplace_back(std::move(combined));
//         }
//       }
//
//       left_dim_pvecs.emplace_back(std::move(left_pvecs));
//       right_dim_pvecs.emplace_back(std::move(right_pvecs));
//     }
//
//     left_bc_parts.emplace_back(std::move(left_dim_pvecs));
//     right_bc_parts.emplace_back(std::move(right_dim_pvecs));
//   }
//
//   //return {left_bc_parts, right_bc_parts};
// }

template<typename P>
std::vector<P> generate_scaled_bc(unscaled_bc_parts<P> const &left_bc_parts,
                                  unscaled_bc_parts<P> const &right_bc_parts,
                                  PDE<P> const &pde,P const time,
                                  std::vector<P> &bc)
{
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

        if (not p_term.left_bc_zero())
        {
          fm::axpy(p_term.left_bc_time_func() ? p_term.left_bc_time_func()(time) : time,
                   left_bc_parts[term_num][dim_num][left_index++], bc);
        }
        if (not p_term.right_bc_zero())
        {
          fm::axpy(p_term.right_bc_time_func() ? p_term.right_bc_time_func()(time) : time,
                   right_bc_parts[term_num][dim_num][right_index++], bc);
        }
      }
    }
  }

  return bc;
}

template<typename P>
std::vector<P>
compute_left_boundary_condition(g_func_type<P> g_func, g_func_type<P> dv_func,
                                P const time, dimension<P> const &dim,
                                vector_func<P> const bc_func)
{
  P const domain_min    = dim.domain_min;
  P const domain_max    = dim.domain_max;
  P const domain_extent = domain_max - domain_min;
  expect(domain_extent > 0);

  int const level  = dim.get_level();
  int const degree = dim.get_degree();

  P const total_cells = fm::ipow2(level);

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = (degree + 1) * total_cells;

  fk::vector<P> bc(dof);

  P g  = g_func ? g_func(domain_min, time) : 1.0;
  P dV = dv_func ? dv_func(domain_min, time) : 1.0;
  if (!std::isfinite(g))
  {
    P const small_dx = domain_per_cell * 1e-7;
    g                = g_func(domain_min + small_dx, time);
    dV               = dv_func(domain_min + small_dx, time);

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    expect(std::isfinite(g));
    expect(std::isfinite(dV));
  }

  /* legendre() returns a 1D matrix - must be converted into a vector */
  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{-1}, degree, legendre_normalization::lin)[0]);

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_min}), time)(0) * g *
                         std::negate{}(dV);

  legendre_polys_at_value.scale(scale_factor);

  fk::vector<P, mem_type::view> destination_slice(bc, 0, degree);

  expect(destination_slice.size() == legendre_polys_at_value.size());

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc.to_std();
}

template<typename P>
std::vector<P>
compute_right_boundary_condition(g_func_type<P> g_func, g_func_type<P> dv_func,
                                 P const time, dimension<P> const &dim,
                                 vector_func<P> const bc_func)
{
  P const domain_min    = dim.domain_min;
  P const domain_max    = dim.domain_max;
  P const domain_extent = domain_max - domain_min;

  expect(domain_extent > 0);

  int const level  = dim.get_level();
  int const degree = dim.get_degree();

  P const total_cells = fm::ipow2(level);

  P const domain_per_cell = domain_extent / total_cells;

  P const dof = (degree + 1) * total_cells;

  fk::vector<P> bc(dof);

  P g  = g_func ? g_func(domain_min, time) : 1.0;
  P dV = dv_func ? dv_func(domain_min, time) : 1.0;
  expect(std::isfinite(dV));
  if (!std::isfinite(g))
  {
    P const small_dx = domain_per_cell * 1e-7;
    g                = g_func(domain_max - small_dx, time);
    dV               = dv_func(domain_max - small_dx, time);

    /* If the above modification was not enough, the choice of g_function
       should be re-evaluated */
    expect(std::isfinite(g));
  }

  fk::vector<P> legendre_polys_at_value = fk::vector<P>(
      legendre(fk::vector<P>{1}, degree, legendre_normalization::lin)[0]);

  P const scale_factor = (1.0 / std::sqrt(domain_per_cell)) *
                         bc_func(fk::vector<P>({domain_max}), time)(0) * g * dV;

  legendre_polys_at_value.scale(scale_factor);

  int const start_index = (degree + 1) * (total_cells - 1);
  int const stop_index  = (degree + 1) * total_cells - 1;
  fk::vector<P, mem_type::view> destination_slice(bc, start_index, stop_index);

  destination_slice = fk::vector<P>(legendre_polys_at_value);

  return bc.to_std();
}

/* explicit instantiations */
#ifdef ASGARD_ENABLE_DOUBLE
template class bc_manager<double>;

template std::array<unscaled_bc_parts<double>, 2> make_unscaled_bc_parts(
    PDE<double> const &pde, elements::table const &table,
    hierarchy_manipulator<double> const &hier, coefficient_matrices<double> &cmats,
    connection_patterns const &,
    int const start_element, int const stop_element, double const t_init = 0);
template std::vector<double> boundary_conditions::generate_scaled_bc(
    unscaled_bc_parts<double> const &left_bc_parts,
    unscaled_bc_parts<double> const &right_bc_parts, PDE<double> const &pde,
    double const time, std::vector<double> &);
template std::vector<double>
boundary_conditions::compute_left_boundary_condition(
    g_func_type<double> g_func, g_func_type<double> dv_func, double const time,
    dimension<double> const &dim, vector_func<double> const bc_func);
template std::vector<double>
boundary_conditions::compute_right_boundary_condition(
    g_func_type<double> g_func, g_func_type<double> dv_func, double const time,
    dimension<double> const &dim, vector_func<double> const bc_func);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template class bc_manager<float>;

template std::array<unscaled_bc_parts<float>, 2>
boundary_conditions::make_unscaled_bc_parts(
    PDE<float> const &pde, elements::table const &table,
    hierarchy_manipulator<float> const &hier, coefficient_matrices<float> &cmats,
    connection_patterns const &,
    int const start_element, int const stop_element, float const t_init = 0);
template std::vector<float> boundary_conditions::generate_scaled_bc(
    unscaled_bc_parts<float> const &left_bc_parts,
    unscaled_bc_parts<float> const &right_bc_parts, PDE<float> const &pde,
    float const time, std::vector<float> &);
template std::vector<float> boundary_conditions::compute_left_boundary_condition(
    g_func_type<float> g_func, g_func_type<float> dv_func, float const time,
    dimension<float> const &dim, vector_func<float> const bc_func);
template std::vector<float>
boundary_conditions::compute_right_boundary_condition(
    g_func_type<float> g_func, g_func_type<float> dv_func, float const time,
    dimension<float> const &dim, vector_func<float> const bc_func);
#endif

} // namespace asgard::boundary_conditions
