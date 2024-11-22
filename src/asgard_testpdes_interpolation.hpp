#pragma once
#include "asgard_interptest_common.hpp"

#ifdef KRON_MODE_GLOBAL

using namespace asgard;

template<typename P>
class nullpde final : public PDE<P>
{
public:
  nullpde(int num_dims, prog_opts const &options,
          int levels = -1, int degree = -1, std::vector<dimension<P>> dims = {})
  {
    int constexpr num_sources       = 0;
    int constexpr num_terms         = 1;
    bool constexpr do_poisson_solve = false;
    // disable implicit steps in IMEX
    bool constexpr do_collision_operator = false;
    bool constexpr has_analytic_soln     = false;

    if (options.start_levels.empty())
    {
      if (levels < 0)
        levels = 2;
    }
    else
    {
      if (levels < 0)
        levels = options.start_levels.front();
    }

    if (not options.degree)
    {
      if (degree < 0)
        degree = 1;
    }
    else
    {
      if (degree < 0)
        degree = options.degree.value();
    }

    if (dims.empty())
      dims.resize(num_dims, dimension<P>(0.0, 1.0, levels, degree, one, nullptr, "x"));

    partial_term<P> pterm = partial_term<P>(
        coefficient_type::mass, negid, nullptr, flux_type::central,
        boundary_condition::periodic, boundary_condition::periodic);

    term<P> fterm(false, "-u", {pterm, }, imex_flag::unspecified);

    term_set<P> terms
        = std::vector<std::vector<term<P>>>{std::vector<term<P>>(num_dims, fterm)};

    this->initialize(options, num_dims, num_sources, num_terms,
                     dims, terms, std::vector<source<P>>{},
                     std::vector<md_func_type<P>>{{}},
                     get_dt_, do_poisson_solve, has_analytic_soln,
                     moment_funcs<P>{}, do_collision_operator);
  }

private:
  static fk::vector<P>
  one(fk::vector<P> const &x, P const = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
        fx[i] = P{1};
    return fx;
  }
  static P negid(P const, P const = 0) { return -1.0; }
  static P get_dt_(dimension<P> const &) { return 1.0; }
};

// creates a projection of series of separable functions
// by setting them as the initial conditions of a nullpde
template<typename P>
discretization_manager<P> make_discretize(
    int num_dims, int levels, int degree, std::vector<std::vector<vector_func<P>>> const &funcs)
{
  std::vector<dimension<P>> dims;
  dims.reserve(num_dims);
  for (int d : indexof<int>(num_dims))
  {
    std::vector<vector_func<P>> f1d;
    for (auto const &f : funcs)
      f1d.push_back(f[d]);
    dims.emplace_back(0, 1, levels, degree, f1d, nullptr, std::to_string(d));
  }

  return discretization_manager<P>(
      std::make_unique<nullpde<P>>(num_dims, prog_opts(), levels, degree, std::move(dims)));
}


// this is part of testing, rename it in case of a conflict
enum class testode_modes
{
  expdecay,
  expexp
};

// 1d problem expressing a system of odes since the solution u(t, x)
// depends only on u(0, x) and no other value within the domain
// the solution is exp(-t) * u(0, x) for PDE (i.e., ode) u_t = - u
//   mockode_modes::expdecay -> u(t, x) = 1
//   mockode_modes::expexp -> u(t, x) = std::exp(x)
// the interp_mode can be on or off so that the simulation can be done
// in two different ways allowing for cross reference of the solutions
template<typename P, bool interp_mode, testode_modes imode>
class testode final : public PDE<P>
{
public:
  testode(prog_opts const &options)
  {
    int constexpr num_dims          = 1;
    int constexpr num_sources       = 0;
    int constexpr num_terms         = (interp_mode) ? 0 : 1;
    bool constexpr do_poisson_solve = false;
    // disable implicit steps in IMEX
    bool constexpr do_collision_operator = false;
    bool constexpr has_analytic_soln     = true;
    int constexpr default_degree         = 1;

    partial_term<P> pterm = partial_term<P>(
        coefficient_type::mass, negid, nullptr, flux_type::central,
        boundary_condition::periodic, boundary_condition::periodic);

    term<P> fterm = term<P>(false, "-u", {pterm,}, imex_flag::unspecified);

    term_set<P> terms;

    if constexpr (interp_mode)
      this->interp_nox_ =
          [](P, std::vector<P> const &u, std::vector<P> &fu)
          -> void {
        for (size_t i = 0; i < u.size(); i++)
          fu[i] = -u[i];
      };
    else
      terms = std::vector<std::vector<term<P>>>{std::vector<term<P>>{fterm,}};

    this->initialize(options, num_dims, num_sources, num_terms,
                     std::vector<dimension<P>>{
                         dimension<P>(0.0, 1.0, 4, default_degree,
                                      component_x, nullptr, "x")},
                     terms, std::vector<source<P>>{},
                     std::vector<md_func_type<P>>{{component_x, component_t}},
                     get_dt_, do_poisson_solve, has_analytic_soln,
                     moment_funcs<P>{}, do_collision_operator);
  }

private:
  static fk::vector<P>
  component_x(fk::vector<P> const &x, P const = 0)
  {
    fk::vector<P> fx(x.size());
    for (int i = 0; i < x.size(); i++)
      if constexpr (imode == testode_modes::expdecay)
        fx[i] = P{1};
      else if constexpr (imode == testode_modes::expexp)
        fx[i] = std::exp(x[i]);
    return fx;
  }

  static P negid(P const, P const = 0) { return -1.0; }

  static fk::vector<P> component_t(fk::vector<P> const &, P const t)
  {
    return {std::exp(-t), };
  }

  static P get_dt_(dimension<P> const &) { return 1.0; }
};

// another enum used only for testing
// can be renamed in case of a conflict
enum class testforcing_modes
{
  interp_exact,
  separable_exact
};

// 2d problem with an exact solution that is the same as continuity_2
// i.e., u_t = - div(u) + f, the initial condition is u(0, x, y) = 0
// the divergence operator is implemented in the regular separable way
// the forcing term is implemented with interpolation
//   testforcing_modes::interp_exact -> interpolated exact solution
//   testforcing_modes::separable_exact -> separable exact solution
template<typename precision, testforcing_modes imode>
class testforcing : public PDE<precision>
{
public:
  using vector = fk::vector<precision>;
  using dimen  = dimension<precision>;

  using partial_term_1d = partial_term<precision>;
  using term_1d         = term<precision>;

  testforcing(prog_opts const &options)
  {
    int constexpr num_dimensions = 2;
    int constexpr num_sources    = 0;
    int constexpr num_terms      = 2;

    bool constexpr do_poisson_solve = false;

    bool constexpr time_independent = false;

    dimen dim0(-1.0, 1.0, 2, 2, zero, nullptr, "x");
    dimen dim1(-2.0, 2.0, 2, 2, zero, nullptr, "y");
    std::vector<dimen> domain = {dim0, dim1};

    partial_term_1d par_der(
        coefficient_type::div, op_coeff, nullptr, flux_type::upwind,
        boundary_condition::periodic, boundary_condition::periodic);

    partial_term_1d par_mass(
        coefficient_type::mass, nullptr, nullptr, flux_type::central,
        boundary_condition::periodic, boundary_condition::periodic);

    term_1d d_x(time_independent, "d_x", {par_der});
    term_1d mass_y(time_independent, "mass_y", {par_mass});

    term_1d mass_x(time_independent, "mass_x", {par_mass});
    term_1d d_y(time_independent, "d_y", {par_der});

    term_set<precision> terms = {
        std::vector<term<precision>>{d_x, mass_y},
        std::vector<term<precision>>{mass_x, d_y}};

    static bool constexpr has_analytic_solution =
        (imode == testforcing_modes::separable_exact);
    std::vector<vector_func<precision>> exact_solution = {};
    if constexpr (has_analytic_solution)
      exact_solution = {exx, exy, ext};

    std::vector<source<precision>> sources = {};

    using P = precision;

    this->interp_x_ = [](P t, vector2d<P> const &nodes, std::vector<P> const &u,
                         std::vector<P> &F)
        -> void {
      for (size_t i = 0; i < u.size(); i++)
      {
        P x  = nodes[i][0];
        P y  = nodes[i][1];
        F[i] = 2 * std::cos(2.0 * t) * std::cos(M_PI * x) * std::sin(2.0 * M_PI * y)
              - M_PI * std::sin(2.0 * t) * std::sin(M_PI * x) * std::sin(2 * M_PI * y)
                + 2 * M_PI * std::sin(2.0 * t) * std::cos(M_PI * x) * std::cos(2 * M_PI * y);
      }
    };

    if constexpr (imode == testforcing_modes::interp_exact)
      this->interp_exact_ = [](P t, vector2d<P> const &x, std::vector<P> &u)
          -> void {
        for (int64_t i = 0; i < x.num_strips(); i++)
          u[i] = std::sin(P{2.0} * t) * std::cos(M_PI * x[i][0])
                * std::sin(P{2.0} * M_PI * x[i][1]);
      };

    this->initialize(
        options, num_dimensions, num_sources, num_terms,
        domain, terms, sources, {exact_solution},
        get_dt, do_poisson_solve, has_analytic_solution);
  }

private:
  static vector exx(vector const &x, precision const = 0)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::cos(M_PI * x[i]);
    return fx;
  }

  static vector exy(vector const &x, precision const = 0)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::sin(precision{2.0} * M_PI * x[i]);
    return fx;
  }

  static vector ext(vector const &, precision const t)
  {
    return {std::sin(precision{2} * t), };
  }

  static vector zero(vector const &x, precision const = 0)
  {
    return vector(x.size());
  }

  static precision get_dt(asgard::dimension<precision> const &dim)
  {
    precision const x_range = dim.domain_max - dim.domain_min;
    precision const dx      = x_range / asgard::fm::ipow2(dim.get_level());
    return dx;
  }

  static precision op_coeff(precision const, precision const)
  {
    return -1.0;
  }
};

// 2d problem with interpolated initial conditions
// u(t, x, y) = sin((x^2 - pi * x)(t+1)) * cos((y^2 - y)(t+1))
// the exact solution is non-separable in time and
// the intial condition is not trivial
// The pde is: u_t = - u_x + f
// allowing for the use of one derivative term (separable) and interp. forcing
// the check for exactness is done against projected exct solution
// note: the main point is that the initial conditions is non-trivial
template<typename precision, bool interp_ic>
class testic : public PDE<precision>
{
public:
  using vector = fk::vector<precision>;
  using dimen  = dimension<precision>;

  using partial_term_1d = partial_term<precision>;
  using term_1d         = term<precision>;

  testic(prog_opts const &options)
  {
    int constexpr num_dimensions = 2;
    int constexpr num_sources    = 0;
    int constexpr num_terms      = 1;

    bool constexpr do_poisson_solve = false;
    bool constexpr time_independent = false;

    std::vector<dimen> domain = {};
    if constexpr (interp_ic)
    {
      // interpolating the initial conditions
      // separable init-cond are set to zero
      dimen dim0(0.0, M_PI, 2, 2, zero, nullptr, "x");
      dimen dim1(0.0, 1.0, 2, 2, zero, nullptr, "y");
      domain = {dim0, dim1};
    }
    else
    {
      // interpolating the initial conditions
      // separable init-cond are set to zero
      dimen dim0(0.0, M_PI, 2, 2, icx, nullptr, "x");
      dimen dim1(0.0, 1.0, 2, 2, icy, nullptr, "y");
      domain = {dim0, dim1};
    }

    partial_term_1d par_der(
        coefficient_type::div, neg_one, nullptr, flux_type::upwind,
        boundary_condition::dirichlet, boundary_condition::dirichlet,
        homogeneity::homogeneous, homogeneity::homogeneous);

    partial_term_1d par_mass(coefficient_type::mass);

    term_1d d_x(time_independent, "d_x", {par_der});
    term_1d mass_y(time_independent, "mass_y", {par_mass});

    term_set<precision> terms = {
        std::vector<term<precision>>{d_x, mass_y},
    };

    bool constexpr has_analytic_solution = true;

    std::vector<vector_func<precision>> exact_solution = {exx, exy};

    std::vector<source<precision>> sources = {};

    using P = precision;

    this->interp_x_ = [](P t, vector2d<P> const &nodes, std::vector<P> const &u,
                         std::vector<P> &F) -> void {
      for (size_t i = 0; i < u.size(); i++)
      {
        P x  = nodes[i][0];
        P y  = nodes[i][1];
        F[i] = xs_dt(x, t) * std::cos(xs(x, t)) * std::cos(yc(y, t))
              - yc_dt(y, t) * std::sin(xs(x, t)) * std::sin(yc(y, t))
                + xs_dx(x, t) * std::cos(xs(x, t)) * std::cos(yc(y, t));
      }
    };

    if constexpr (interp_ic)
      this->interp_initial_ = [](vector2d<P> const &nodes, std::vector<P> &u)
          -> void {
        for (int64_t i = 0; i < nodes.num_strips(); i++)
        {
          P x  = nodes[i][0];
          P y  = nodes[i][1];
          u[i] = std::sin(xs(x, 0)) * std::cos(yc(y, 0));
        }
      };

    this->initialize(
        options, num_dimensions, num_sources, num_terms,
        domain, terms, sources, {exact_solution},
        get_dt, do_poisson_solve, has_analytic_solution);
  }

private:
  // expression that goes inside the sin, i.e., sin(xs(x))
  static precision xs(precision x, precision t)
  {
    return (x * x - M_PI * x) * (t + 1);
  }
  static precision xs_dx(precision x, precision t) // drivative in x
  {
    return (2 * x - M_PI) * (t + 1);
  }
  static precision xs_dt(precision x, precision) // drivative in t
  {
    return (x * x - M_PI * x);
  }

  // expression that goes inside the cos, i.e., cos(yc(y))
  static precision yc(precision y, precision t)
  {
    return -M_PI / 2 + (y * y - y) * (t + 1);
  }
  static precision yc_dt(precision y, precision)
  {
    return (y * y - y);
  }

  static vector icx(vector const &x, precision const = 0)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::sin(xs(x[i], 0));
    return fx;
  }
  static vector icy(vector const &y, precision const = 0)
  {
    vector fy(y.size());
    for (int i = 0; i < y.size(); i++)
      fy[i] = std::cos(yc(y[i], 0));
    return fy;
  }
  static vector exx(vector const &x, precision const t)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::sin(xs(x[i], t));
    return fx;
  }
  static vector exy(vector const &y, precision const t)
  {
    vector fy(y.size());
    for (int i = 0; i < y.size(); i++)
      fy[i] = std::cos(yc(y[i], t));
    return fy;
  }

  static vector zero(vector const &x, precision const = 0)
  {
    return vector(x.size());
  }

  static precision get_dt(asgard::dimension<precision> const &dim)
  {
    precision const x_range = dim.domain_max - dim.domain_min;

    precision const dx = x_range / asgard::fm::ipow2(dim.get_level());
    return dx;
  }

  static precision neg_one(precision const, precision const)
  {
    return -1.0;
  }
};

// for a pde with existing exact solution (separable or not)
// simulates the pde and returns the list of rmse at each time step
template<typename pde_type>
auto time_advance_errors(prog_opts const &opts)
{
  using P = typename pde_type::precision_mode;

  prog_opts silent_opts = opts;

  silent_opts.ignore_exact = true;

  discretization_manager<P> disc(std::make_unique<pde_type>(silent_opts));

  auto const &pde = disc.get_pde();

  expect(pde.has_analytic_soln() or pde.interp_exact());

  auto const num_ranks = get_num_ranks();
  if (num_ranks > 1) // not interpolation for MPI yet
    return std::vector<P>{};

  int64_t const num_final = disc.final_time_step();

  std::vector<P> errors;
  errors.reserve(num_final);

  // -- time loop
  for (auto i : indexof(disc.final_time_step()))
  {
    ignore(i);

    advance_time(disc, 1);

    auto rmse = disc.rmse_exact_sol();
    rassert(rmse, "could not compute exact solution");
    errors.push_back((*rmse)[0][0]);
  }

  return errors;
}

#endif
