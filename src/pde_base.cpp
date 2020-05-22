#include "pde_base.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "transformations.hpp"
#include <numeric>

template<typename P>
void PDE<P>::regenerate_coefficients(P const time, bool const rotate)
{
  for (int i = 0; i < num_dims; ++i)
  {
    dimension<P> const &dim = get_dimensions()[i];

    for (int j = 0; j < num_terms; ++j)
    {
      term<P> const &term_1D = get_terms()[j][i];

      std::vector<partial_term<P>> const &partial_terms =
          term_1D.get_partial_terms();

      /* generate the first partial term */
      fk::matrix<double> term_coeff =
          generate_coefficients(dim, term_1D, partial_terms[0], time, rotate);

      /* set the partial term's coefficient matrix */
      set_partial_coefficients(j, i, 0, fk::matrix<P>(term_coeff));

      for (int k = 1; k < static_cast<int>(partial_terms.size()); ++k)
      {
        fk::matrix<double> partial_term_coeff =
            generate_coefficients(dim, term_1D, partial_terms[k], time, rotate);

        term_coeff = term_coeff * partial_term_coeff;

        set_partial_coefficients(j, i, k, fk::matrix<P>(partial_term_coeff));
      }

      set_coefficients(fk::matrix<P>(term_coeff), j, i);
    }
  }
}

template<typename P>
fk::matrix<double>
PDE<P>::generate_coefficients(dimension<P> const &dim, term<P> const &term_1D,
                              partial_term<P> const &pterm, P const time,
                              bool const rotate)
{
  assert(time >= 0.0);
  // setup jacobi of variable x and define coeff_mat
  int const num_points = fm::two_raised_to(dim.get_level());
  // note that grid_spacing is the symbol typically reserved for grid spacing
  double const grid_spacing    = (dim.domain_max - dim.domain_min) / num_points;
  int const degrees_freedom_1d = dim.get_degree() * num_points;
  fk::matrix<double> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // get quadrature points and quadrature_weights.
  // we do the two-step store because we cannot have 'static' bindings
  static auto const legendre_values =
      legendre_weights<double>(dim.get_degree(), -1.0, 1.0);
  auto const [quadrature_points, quadrature_weights] = legendre_values;
  auto const [legendre_poly_L, legendre_poly_R]      = [&]() {
    auto [lP_L, lPP_L] = legendre(fk::vector<double>{-1}, dim.get_degree());
    lP_L               = lP_L * (1 / std::sqrt(grid_spacing));
    auto [lP_R, lPP_R] = legendre(fk::vector<double>{+1}, dim.get_degree());
    lP_R               = lP_R * (1 / std::sqrt(grid_spacing));
    // this is to get around unused warnings (until c++20)
    auto const ignore = [](auto ignored) { (void)ignored; };
    ignore(lPP_L);
    ignore(lPP_R);
    return std::array<fk::matrix<double>, 2>{lP_L, lP_R};
  }();

  auto const legendre_poly_L_t =
      fk::matrix<double>(legendre_poly_L).transpose();
  auto const legendre_poly_R_t =
      fk::matrix<double>(legendre_poly_R).transpose();

  // get the basis functions and derivatives for all k
  // this auto is std::array<fk::matrix<P>, 2>
  auto const [legendre_poly,
              legendre_prime] = [&, quadrature_points = quadrature_points]() {
    auto [lP, lPP] = legendre(quadrature_points, dim.get_degree());

    lP  = lP * (1.0 / std::sqrt(grid_spacing));
    lPP = lPP * (1.0 / std::sqrt(grid_spacing) * 2.0 / grid_spacing);

    return std::array<fk::matrix<double>, 2>{lP, lPP};
  }();

  auto const legendre_poly_t  = fk::matrix<double>(legendre_poly).transpose();
  auto const legendre_prime_t = fk::matrix<double>(legendre_prime).transpose();

  // get jacobian
  auto const jacobi = grid_spacing / 2;

  // convert term input data from wavelet space to realspace
  fk::matrix<double> const forward_trans_transpose =
      dim.get_from_basis_operator();
  fk::vector<double> const data = fk::vector<double>(term_1D.get_data());
  fk::vector<double> const data_real =
      forward_trans_transpose * fk::vector<double>(term_1D.get_data());

  for (int i = 0; i < num_points; ++i)
  {
    // get left and right locations for this element
    auto const x_left  = dim.domain_min + i * grid_spacing;
    auto const x_right = x_left + grid_spacing;

    // get index for current, firs and last element
    int const current = dim.get_degree() * i;
    int const first   = 0;
    int const last    = dim.get_degree() * (num_points - 1);

    // map quadrature points from [-1,1] to physical domain of this i element
    fk::vector<double> const quadrature_points_i =
        [&, quadrature_points = quadrature_points]() {
          fk::vector<double> quadrature_points_copy = quadrature_points;
          std::transform(
              quadrature_points_copy.begin(), quadrature_points_copy.end(),
              quadrature_points_copy.begin(), [&](double const elem) {
                return ((elem + 1) / 2 + i) * grid_spacing + dim.domain_min;
              });
          return quadrature_points_copy;
        }();

    fk::vector<double> const g_func = [&, legendre_poly = legendre_poly]() {
      // get realspace data at quadrature points
      // NOTE : this is unused pending updating G functions to accept "dat"
      fk::vector<double> data_real_quad =
          legendre_poly *
          fk::vector<double, mem_type::const_view>(
              data_real, current, current + dim.get_degree() - 1);
      // get g(x,t,dat)
      // FIXME : add dat as a argument to the G functions
      fk::vector<double> g(quadrature_points_i.size());
      for (int i = 0; i < quadrature_points_i.size(); ++i)
      {
        g(i) = pterm.g_func(quadrature_points_i(i), time);
      }
      return g;
    }();

    auto const block = [&, legendre_poly = legendre_poly,
                        quadrature_weights = quadrature_weights]() {
      fk::matrix<double> tmp(legendre_poly.nrows(), legendre_poly.ncols());

      for (int i = 0; i <= tmp.nrows() - 1; i++)
      {
        for (int j = 0; j <= tmp.ncols() - 1; j++)
        {
          tmp(i, j) =
              g_func(i) * legendre_poly(i, j) * quadrature_weights(i) * jacobi;
        }
      }
      fk::matrix<double> block(dim.get_degree(), dim.get_degree());

      if (pterm.coeff_type == coefficient_type::mass)
      {
        block = legendre_poly_t * tmp;
      }
      else if (pterm.coeff_type == coefficient_type::grad)
      {
        block = legendre_prime_t * tmp * (-1);
      }
      return block;
    }();

    // set the block at the correct position
    fk::matrix<double> const curr_block =
        fk::matrix<double, mem_type::view>(
            coefficients, current, current + dim.get_degree() - 1, current,
            current + dim.get_degree() - 1) +
        block;
    coefficients.set_submatrix(current, current, curr_block);

    if (pterm.coeff_type == coefficient_type::grad)
    {
      // setup numerical flux choice/boundary conditions
      //
      // - <funcCoef*{q},p>
      //----------------------------------------------
      // Numerical Flux is defined as
      // Flux = {{f}} + C/2*[[u]]
      //      = ( f_L + f_R )/2 + FunCoef*( u_R - u_L )/2
      // [[v]] = v_R - v_L

      // FIXME G functions should accept G(x,p,t,dat), since we don't know how
      // the dat is going to be used in the G function (above it is used as
      // linear multuplication but this is not always true)

      auto const flux_left  = pterm.g_func(x_left, time);
      auto const flux_right = pterm.g_func(x_right, time);

      // get the "trace" values
      // (values at the left and right of each element for all k)
      auto trace_value_1 =
          (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
          (legendre_poly_L_t * legendre_poly_R) *
              (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      auto trace_value_2 =
          (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
          (legendre_poly_L_t * legendre_poly_L) *
              (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
      auto trace_value_3 =
          (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
          (legendre_poly_R_t * legendre_poly_R) *
              (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
      auto trace_value_4 =
          (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
          (legendre_poly_R_t * legendre_poly_L) *
              (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);

      // If dirichelt
      // u^-_LEFT = g(LEFT)
      // u^+_RIGHT = g(RIGHT)

      if (pterm.left == boundary_condition::dirichlet) // left dirichlet
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_2 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (pterm.right == boundary_condition::dirichlet) // right dirichlet
      {
        if (i == num_points - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_3 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      // If neumann
      // (gradient u)*num_points = g
      // by splitting grad u = q by LDG methods, the B.C is changed to
      // q*num_points = g (=> q = g for 1D variable)
      // only work for derivatives greater than 1

      if (pterm.left == boundary_condition::neumann) // left neumann
      {
        if (i == 0)
        {
          trace_value_1 =
              (legendre_poly_L_t * (legendre_poly_L - legendre_poly_L)) * (-1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
          trace_value_4 =
              (legendre_poly_R_t * legendre_poly_L) * (+1 * flux_right / 2) +
              (legendre_poly_R_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_right) / 2 * +1);
        }
      }

      if (pterm.right == boundary_condition::neumann) // right neumann
      {
        if (i == num_points - 1)
        {
          trace_value_1 =
              (legendre_poly_L_t * legendre_poly_R) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_R) *
                  (+1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_2 =
              (legendre_poly_L_t * legendre_poly_L) * (-1 * flux_left / 2) +
              (legendre_poly_L_t * legendre_poly_L) *
                  (-1 * pterm.get_flux_scale() * std::abs(flux_left) / 2 * -1);
          trace_value_3 =
              (legendre_poly_R_t * legendre_poly_R) * (+1 * flux_right);
          trace_value_4 =
              (legendre_poly_R_t * (legendre_poly_R - legendre_poly_R)) * (+1);
        }
      }

      if (pterm.coeff_type == coefficient_type::grad)
      {
        // Add trace values to matrix

        int row1 = current;
        int col1 = current - dim.get_degree();

        int row2 = current;
        int col2 = current;

        int row3 = current;
        int col3 = current;

        int row4 = current;
        int col4 = current + dim.get_degree();

        if (pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          if (i == 0)
          {
            row1 = current;
            col1 = last;
          }
          if (i == num_points - 1)
          {
            row4 = current;
            col4 = first;
          }
        }

        if (i != 0 || pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          // Add trace part 1
          fk::matrix<double, mem_type::view> block1(
              coefficients, row1, row1 + dim.get_degree() - 1, col1,
              col1 + dim.get_degree() - 1);
          block1 = block1 + trace_value_1;
        }
        // Add trace part 2
        fk::matrix<double, mem_type::view> block2(
            coefficients, row2, row2 + dim.get_degree() - 1, col2,
            col2 + dim.get_degree() - 1);
        block2 = block2 + trace_value_2;

        // Add trace part 3
        fk::matrix<double, mem_type::view> block3(
            coefficients, row3, row3 + dim.get_degree() - 1, col3,
            col3 + dim.get_degree() - 1);
        block3 = block3 + trace_value_3;

        if (i != num_points - 1 || pterm.left == boundary_condition::periodic ||
            pterm.right == boundary_condition::periodic)
        {
          // Add trace part 4
          fk::matrix<double, mem_type::view> block4(
              coefficients, row4, row4 + dim.get_degree() - 1, col4,
              col4 + dim.get_degree() - 1);
          block4 = block4 + trace_value_4;
        }
      }
    }
  }
  if (rotate)
  {
    // transform matrix to wavelet space
    fk::matrix<double> const forward_trans = dim.get_to_basis_operator();

    // These apply_*_fmwt() routines do the following operation:
    // coefficients = forward_trans * coefficients * forward_trans_transpose;
    coefficients = apply_right_fmwt_transposed(
        forward_trans,
        apply_left_fmwt(forward_trans, coefficients, dim.get_degree(),
                        dim.get_level()),
        dim.get_degree(), dim.get_level());
  }
  return coefficients;
}

template<typename P>
PDE<P>::PDE(int const num_levels, int const degree, int const num_dims,
            int const num_sources, int const num_terms,
            std::vector<dimension<P>> const dimensions, term_set<P> const terms,
            std::vector<source<P>> const sources,
            std::vector<vector_func<P>> const exact_vector_funcs,
            scalar_func<P> const exact_time, dt_func<P> const get_dt,
            bool const do_poisson_solve, bool const has_analytic_soln)
    : num_dims(num_dims), num_sources(num_sources), num_terms(num_terms),
      sources(sources), exact_vector_funcs(exact_vector_funcs),
      exact_time(exact_time), do_poisson_solve(do_poisson_solve),
      has_analytic_soln(has_analytic_soln), dimensions_(dimensions),
      terms_(terms)
{
  assert(num_dims > 0);
  assert(num_sources >= 0);
  assert(num_terms > 0);

  assert(dimensions.size() == static_cast<unsigned>(num_dims));
  assert(terms.size() == static_cast<unsigned>(num_terms));
  assert(sources.size() == static_cast<unsigned>(num_sources));

  for (auto tt : terms)
  {
    for (auto t : tt)
    {
      std::vector<partial_term<P>> const &pterms = t.get_partial_terms();
      for (auto p : pterms)
      {
        if (p.left_homo == homogeneity::homogeneous)
          assert(static_cast<int>(p.left_bc_funcs.size()) == 0);
        else if (p.left_homo == homogeneity::inhomogeneous)
          assert(static_cast<int>(p.left_bc_funcs.size()) == num_dims);

        if (p.right_homo == homogeneity::homogeneous)
          assert(static_cast<int>(p.right_bc_funcs.size()) == 0);
        else if (p.right_homo == homogeneity::inhomogeneous)
          assert(static_cast<int>(p.right_bc_funcs.size()) == num_dims);
      }
    }
  }

  // ensure analytic solution functions were provided if this flag is set
  if (has_analytic_soln)
  {
    assert(exact_vector_funcs.size() == static_cast<unsigned>(num_dims));
  }

  // check all terms
  for (std::vector<term<P>> const &term_list : terms_)
  {
    assert(term_list.size() == static_cast<unsigned>(num_dims));

    for (term<P> const &term_1D : term_list)
    {
      assert(term_1D.get_partial_terms().size() > 0);
    }
  }

  // modify for appropriate level/degree
  // if default lev/degree not used
  if (num_levels != -1 || degree != -1)
  {
    // FIXME eventually independent levels for each dim will be
    // supported
    for (dimension<P> &d : dimensions_)
    {
      if (num_levels != -1)
      {
        assert(num_levels > 1);
        d.set_level(num_levels);
      }
      if (degree != -1)
      {
        assert(degree > 0);
        d.set_degree(degree);
      }
    }

    for (std::vector<term<P>> &term_list : terms_)
    {
      // positive, bounded size - safe compare
      for (int i = 0; i < static_cast<int>(term_list.size()); ++i)
      {
        term_list[i].set_data(dimensions_[i], fk::vector<P>());
        term_list[i].set_coefficients(
            dimensions_[i],
            eye<P>(term_list[i].degrees_freedom(dimensions_[i])));
      }
    }
  }
  // check all dimensions
  for (dimension<P> const d : dimensions_)
  {
    assert(d.get_degree() > 0);
    assert(d.get_level() > 1);
    assert(d.domain_max > d.domain_min);
  }

  // check all sources
  for (source<P> const s : this->sources)
  {
    assert(s.source_funcs.size() == static_cast<unsigned>(num_dims));
  }

  // set the dt
  dt_ = get_dt(dimensions_[0]);

  /* generate coefficients based on default parameters */
  regenerate_coefficients();
}

/* Explicit instantiations */
template class PDE<float>;
template class PDE<double>;
