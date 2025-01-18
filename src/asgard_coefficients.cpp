#include "asgard_coefficients.hpp"

#include "asgard_coefficients_mats.hpp"

namespace asgard
{

template<typename P>
void gen_flux_matrix(
    dimension<P> const &dim, partial_term<P> const &pterm,
    int const level, P const time, block_tri_matrix<P> &coefficients)
{
  switch (pterm.coeff_type())
  {
  case coefficient_type::mass:
    throw std::runtime_error("trying to generate block_tri_matrix from pterm with no flux in the coefficient");
    break;
  case coefficient_type::grad:
    gen_tri_cmat<P, coefficient_type::grad>(dim, pterm, level, time, coefficients);
    break;
  case coefficient_type::div:
    gen_tri_cmat<P, coefficient_type::div>(dim, pterm, level, time, coefficients);
    break;
  default: // case coefficient_type::penalty:
    gen_tri_cmat<P, coefficient_type::penalty>(dim, pterm, level, time, coefficients);
    break;
  };
}

template<typename P>
void gen_mass_matrix(
    coupled_term_data<P> const &edata, dimension<P> const &dim, partial_term<P> const &pterm,
    int const level, P const time, block_diag_matrix<P> &coefficients)
{
  expect(pterm.coeff_type() == coefficient_type::mass);
  expect(not has_flux(pterm.coeff_type()));
  // add a case statement if there are other coefficient_type instances with no flux
  // checking the functor type
  // the regular case is to use pterm.gfunc * pterm.dv, if either is missing, assume it's 1
  // we can also handle cases of dependence on external data
  switch(pterm.depends())
  {
    case pterm_dependence::lenard_bernstein_coll_theta_1x1v:
      gen_diag_mom_by_mom0<P, 1, pterm_dependence::lenard_bernstein_coll_theta_1x1v>(
          dim, pterm, level, time, edata.moments, coefficients);
      break;
    case pterm_dependence::lenard_bernstein_coll_theta_1x2v:
      gen_diag_mom_by_mom0<P, 1, pterm_dependence::lenard_bernstein_coll_theta_1x2v>(
          dim, pterm, level, time, edata.moments, coefficients);
      break;
    case pterm_dependence::lenard_bernstein_coll_theta_1x3v:
      gen_diag_mom_by_mom0<P, 1, pterm_dependence::lenard_bernstein_coll_theta_1x3v>(
          dim, pterm, level, time, edata.moments, coefficients);
      break;
    case pterm_dependence::moment_divided_by_density:
      if (pterm.mom_index() > 0) {
        gen_diag_mom_by_mom0<P, 1, pterm_dependence::moment_divided_by_density>(
            dim, pterm, level, time, edata.moments, coefficients);
      } else {
        gen_diag_mom_by_mom0<P, -1, pterm_dependence::moment_divided_by_density>(
            dim, pterm, level, time, edata.moments, coefficients);
      }
      break;
    case pterm_dependence::electric_field:
      expect(edata.electric_field.size() == static_cast<size_t>(fm::ipow2(level)));
      if (pterm.g_func_f() and pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int i, P const x, P const t)->P{
                return pterm.g_func_f()(x, t, edata.electric_field[i]) * pterm.dv_func()(x, t);
            }, coefficients);
      } else if (pterm.g_func_f()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int i, P const x, P const t)->P{
                return pterm.g_func_f()(x, t, edata.electric_field[i]);
            }, coefficients);
      } else if (pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int i, P const x, P const t)->P{ return edata.electric_field[i] * pterm.dv_func()(x, t); },
            coefficients);
      } else {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int i, P const, P const)-> P{ return edata.electric_field[i]; }, coefficients);
      }
      break;
    case pterm_dependence::electric_field_infnrm:
      expect(edata.electric_field.size() == static_cast<size_t>(fm::ipow2(level)));
      if (pterm.g_func_f() and pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&, mE = edata.electric_field_infnrm.value()](int, P const x, P const t)->P{
                return pterm.g_func_f()(x, t, mE) * pterm.dv_func()(x, t);
            }, coefficients);
      } else if (pterm.g_func_f()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&, mE = edata.electric_field_infnrm.value()](int, P const x, P const t)->P{
                return pterm.g_func_f()(x, t, mE);
            }, coefficients);
      } else if (pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&, mE = edata.electric_field_infnrm.value()](int, P const x, P const t)->P{
                return mE * pterm.dv_func()(x, t);
            },
            coefficients);
      } else {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [mE = edata.electric_field_infnrm.value()](int, P const, P const)-> P{
                return mE; }, coefficients);
      }
      break;
    default: // case pterm_dependence::none:
      if (pterm.g_func() and pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int, P const x, P const t)->P{ return pterm.g_func()(x, t) * pterm.dv_func()(x, t); },
            coefficients);
      } else if (pterm.g_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int, P const x, P const t)->P{ return pterm.g_func()(x, t); },
            coefficients);
      } else if (pterm.dv_func()) {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int, P const x, P const t)->P{ return pterm.dv_func()(x, t); },
            coefficients);
      } else {
        gen_diag_cmat<P, coefficient_type::mass>(dim, level, time,
            [&](int, P const, P const)-> P{ return 1.0; }, coefficients);
      }
      break;
  }
}

template<typename P>
void generate_coefficients(
    PDE<P> &pde, coefficient_matrices<P> &mats, connection_patterns const &conn,
    hierarchy_manipulator<P> const &hier, P const time, coeff_update_mode mode)
{
  tools::time_event time_generating_("gen_coefficients");
  expect(time >= 0.0);

  static block_tri_matrix<P> raw_tri;
  static block_diag_matrix<P> raw_diag;

  int const num_dims = pde.num_dims();
  int const pdof     = hier.degree() + 1;

  for (int d : indexof<int>(num_dims))
  {
    dimension<P> const &dim = pde.get_dimensions()[d];

    int const level = dim.get_level();

    for (int t : indexof<int>(pde.num_terms()))
    {
      auto const &term1d = pde.get_terms()[t][d];
      auto const &pterms = term1d.get_partial_terms();

      switch (mode) {
        case coeff_update_mode::imex_explicit:
          if (term1d.flag() != imex_flag::imex_explicit)
            continue;
          break;
        case coeff_update_mode::imex_implicit:
          if (term1d.flag() != imex_flag::imex_implicit)
            continue;
          break;
        case coeff_update_mode::poisson:
          if (not term1d.has_dependence(pterm_dependence::electric_field))
            continue;
          break;
        case coeff_update_mode::independent:
          // update only the terms that don't have moment dependence
          if (not term1d.is_moment_independant())
            continue;
          break;
        default: // case coeff_update_mode::all, do not skip anything
          break;
      };

      // do not recompute coefficients that are constant in time
      // and have already been computed for the given level (or above)
      if (not term1d.time_dependent() and mats.current_levels[d][t] == level) {
        continue; // we have what we need
      }

      mats.current_levels[d][t] = level; // computing for this level

      expect(pterms.size() >= 1);
      if (pterms.size() == 1)
      {
        // the single term case uses less scratch space
        generate_partial_mass(d, dim, pterms[0], hier, time,
                              mats.pterm_mass[t * num_dims + d][0]);

        if (has_flux(pterms[0].coeff_type()))
        {
          gen_flux_matrix<P>(dim, pterms[0], level, time, raw_tri);

          if (mats.pterm_mass[t * num_dims + d][0].has_level(level))
            invert_mass(pdof, mats.pterm_mass[t * num_dims + d][0][level], raw_tri);

          mats.term_coeffs[t * num_dims + d] = hier.tri2hierarchical(raw_tri, level, conn);
        }
        else // no-flux, e.g., mass matrix
        {
          gen_mass_matrix<P>(mats.edata, dim, pterms[0], level, time, raw_diag);

          if (mats.pterm_mass[t * num_dims + d][0].has_level(level))
            invert_mass(pdof, mats.pterm_mass[t * num_dims + d][0][level], raw_diag);

          mats.term_coeffs[t * num_dims + d] = hier.diag2hierarchical(raw_diag, level, conn);
        }
      }
      else
      {
        // additional scratch space matrices
        static block_tri_matrix<P> raw_tri0, raw_tri1;
        static block_diag_matrix<P> raw_diag0, raw_diag1;

        // switch to non-owning pointers for easier and cheaper swapping
        // will hold the final result
        block_tri_matrix<P> *rtri = &raw_tri;
        block_diag_matrix<P> *rdiag = &raw_diag;

        // used for scratch space
        block_tri_matrix<P> *rtri0 = &raw_tri0;
        block_tri_matrix<P> *rtri1 = &raw_tri1;
        block_diag_matrix<P> *rdiag0 = &raw_diag0;
        block_diag_matrix<P> *rdiag1 = &raw_diag1;

        // check if using diagonal or tri-diagonal structure
        bool is_tri_diagonal = false;
        for (auto const &p : pterms)
        {
          if (has_flux(p.coeff_type()))
          {
            is_tri_diagonal = true;
            break;
          }
        }

        if (is_tri_diagonal)
        {
          for (auto fi : indexof(pterms.size()))
          {
            // looping over partial-terms in reverse order
            int64_t const ip = static_cast<int64_t>(pterms.size()) - fi - 1;
            auto const &pterm = pterms[ip];
            generate_partial_mass(d, dim, pterm, hier, time,
                                  mats.pterm_mass[t * num_dims + d][fi]);

            if (fi == 0)
            {
              if (has_flux(pterm.coeff_type()))
              {
                gen_flux_matrix<P>(dim, pterm, level, time, *rtri);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rtri);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.tri2hierarchical(*rtri, level, conn);
              }
              else
              {
                gen_mass_matrix<P>(mats.edata, dim, pterm, level, time, *rdiag);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag);

                *rtri = *rdiag;
                mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag, level, conn);
              }
            }
            else
            {
              if (has_flux(pterm.coeff_type()))
              {
                gen_flux_matrix<P>(dim, pterm, level, time, *rtri0);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rtri0);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.tri2hierarchical(*rtri0, level, conn);

                if (rtri1->nblock() != rtri->nblock() or rtri1->nrows() != rtri->nrows())
                  rtri1->resize_and_zero(*rtri);

                std::swap(rtri, rtri1);
                gemm_block_tri(pdof, *rtri0, *rtri1, *rtri);
              }
              else
              {
                gen_mass_matrix<P>(mats.edata, dim, pterm, level, time, *rdiag0);

                if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag0);

                mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag0, level, conn);

                if (rtri1->nblock() != rtri->nblock() or rtri1->nrows() != rtri->nrows())
                  rtri1->resize_and_zero(*rtri);

                std::swap(rtri, rtri1);
                gemm_diag_tri(pdof, *rdiag0, *rtri1, *rtri);
              }
            }
          }
          mats.term_coeffs[t * num_dims + d] = hier.tri2hierarchical(*rtri, level, conn);
        }
        else
        {
          // using a series of diagonal matrices
          for (auto fi : indexof(pterms.size()))
          {
            // looping over partial-terms in reverse order
            int64_t const ip = static_cast<int64_t>(pterms.size()) - fi - 1;
            auto const &pterm = pterms[ip];

            generate_partial_mass(d, dim, pterm, hier, time,
                                  mats.pterm_mass[t * num_dims + d][fi]);

            if (fi == 0)
            {
              gen_mass_matrix<P>(mats.edata, dim, pterm, level, time, *rdiag);

              if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag);

              mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag, level, conn);
            }
            else
            {
              gen_mass_matrix<P>(mats.edata, dim, pterm, level, time, *rdiag0);

              if (mats.pterm_mass[t * num_dims + d][fi].has_level(level))
                  invert_mass(pdof, mats.pterm_mass[t * num_dims + d][fi][level], *rdiag0);

              mats.pterm_coeffs[t * num_dims + d][ip] = hier.diag2hierarchical(*rdiag0, level, conn);

              if (rdiag1->nblock() != rdiag0->nblock() or rdiag1->nrows() != rdiag0->nrows())
                rdiag1->resize_and_zero(*rdiag0);

              std::swap(rdiag, rdiag1);
              gemm_block_diag(pdof, *rdiag0, *rdiag1, *rdiag);
            }
          }
          mats.term_coeffs[t * num_dims + d] = hier.diag2hierarchical(*rdiag, level, conn);
        }
      } // if multiple partial terms
    } // for num-terms
  } // for num-dims
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims(); ++d)
  {
    indices(d) = coords(d) * (degree + 1);
  }
  return indices;
}
template<typename P>
void build_system_matrix(
    PDE<P> const &pde, std::function<fk::matrix<P>(int, int)> get_coeffs,
    elements::table const &elem_table, fk::matrix<P> &A,
    element_subgrid const &grid, imex_flag const imex)
{
  // assume uniform degree for now
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = fm::ipow(degree + 1, pde.num_dims());

  int const A_cols = elem_size * grid.ncols();
  int const A_rows = elem_size * grid.nrows();
  expect(A.ncols() == A_cols && A.nrows() == A_rows);

  using key_type = std::pair<int, int>;
  using val_type = fk::matrix<P, mem_type::owner, resource::host>;
  std::map<key_type, val_type> coef_cache;

  // copy coefficients to host for subsequent use
  for (int k = 0; k < pde.num_terms(); ++k)
  {
    for (int d = 0; d < pde.num_dims(); d++)
    {
      coef_cache.emplace(key_type(k, d), get_coeffs(k, d));
    }
  }

  auto terms = pde.get_terms();

  // loop over elements
  for (auto i = grid.row_start; i <= grid.row_stop; ++i)
  {
    // first, get linearized indices for this element
    //
    // calculate from the level/cell indices for each
    // dimension
    fk::vector<int> const coords = elem_table.get_coords(i);
    expect(coords.size() == pde.num_dims() * 2);
    fk::vector<int> const elem_indices = linearize(coords);

    int const global_row = i * elem_size;

    // calculate the row portion of the
    // operator position used for this
    // element's gemm calls
    fk::vector<int> const operator_row =
        linear_coords_to_indices(pde, degree, elem_indices);

    // loop over connected elements. for now, we assume
    // full connectivity
    for (int j = grid.col_start; j <= grid.col_stop; ++j)
    {
      // get linearized indices for this connected element
      fk::vector<int> const coords_nD = elem_table.get_coords(j);
      expect(coords_nD.size() == pde.num_dims() * 2);
      fk::vector<int> const connected_indices = linearize(coords_nD);

      // calculate the col portion of the
      // operator position used for this
      // element's gemm calls
      fk::vector<int> const operator_col =
          linear_coords_to_indices(pde, degree, connected_indices);

      for (int k = 0; k < pde.num_terms(); ++k)
      {
        std::vector<fk::matrix<P>> kron_vals;
        fk::matrix<P> kron0(1, 1);
        // if using imex, include only terms that match the flag
        if (imex == imex_flag::unspecified || terms[k][0].flag() == imex)
        {
          kron0(0, 0) = 1.0;
        }
        else
        {
          kron0(0, 0) = 0.0;
        }
        kron_vals.push_back(std::move(kron0));
        for (int d = 0; d < pde.num_dims(); d++)
        {
          fk::matrix<P, mem_type::view> op_view = fk::matrix<P, mem_type::view>(
              coef_cache[key_type(k, d)], operator_row(d),
              operator_row(d) + degree, operator_col(d),
              operator_col(d) + degree);
          fk::matrix<P> k_new = kron_vals[d].kron(op_view);
          kron_vals.push_back(std::move(k_new));
        }

        // calculate the position of this element in the
        // global system matrix
        int const global_col = j * elem_size;
        auto const &k_tmp    = kron_vals.back();
        fk::matrix<P, mem_type::view> A_view(
            A, global_row - grid.row_start * elem_size,
            global_row + k_tmp.nrows() - 1 - grid.row_start * elem_size,
            global_col - grid.col_start * elem_size,
            global_col + k_tmp.ncols() - 1 - grid.col_start * elem_size);

        A_view = A_view + k_tmp;
      }
    }
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template void generate_coefficients<double>(
    PDE<double> &, coefficient_matrices<double> &, connection_patterns const &,
    hierarchy_manipulator<double> const &, double, coeff_update_mode);

template void build_system_matrix<double>(
    PDE<double> const &, std::function<fk::matrix<double>(int, int)>,
    elements::table const &, fk::matrix<double> &,
    element_subgrid const &, imex_flag const);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template void generate_coefficients<float>(
    PDE<float> &, coefficient_matrices<float> &, connection_patterns const &,
    hierarchy_manipulator<float> const &, float, coeff_update_mode);

template void build_system_matrix<float>(
    PDE<float> const &, std::function<fk::matrix<float>(int, int)>,
    elements::table const &, fk::matrix<float> &,
    element_subgrid const &, imex_flag const);
#endif

} // namespace asgard
