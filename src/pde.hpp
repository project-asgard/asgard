#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

//
// The choices for supported PDE types
//
enum class PDE_opts
{
  vlasov4,  // PDE corresponding to Fig. 4 in FIXME
  vlasov43, // PDE corresponding to Fig. 4.3 in FIXME
  vlasov5,  // PDE corresponding to Fig. 5 in FIXME
  vlasov7,  // PDE corresponding to Fig. 7 in FIXME
  vlasov8,  // PDE corresponding to Fig. 8 in FIXME
  pde_user  // FIXME will need to add the user supplied PDE choice
};

//
// for passing around vector/scalar functions used by the PDE
//
template<typename P>
using vector_func = std::function<std::vector<P>(std::vector<P> const)>;
template<typename P>
using scalar_func = std::function<P(P const)>;

// ---------------------------------------------------------------------------
//
// abstract base class defining interface for PDEs
//
// ----------------------------------------------------------------------------
template<typename P>
class PDE
{
public:
  // FIXME we will need these eventually
  // std::vector<Term<P>> const terms;

  // clang-format off
  PDE(int const num_dims,
      int const num_sources,
      int const num_terms,
      P const domain_mins[], // the mininimum grid value in each dimension
      P const domain_maxs[], // the maximum grid value in each dimension
      bool const do_poisson_solve,
      bool const has_analytic_soln = false)

      : num_dims(num_dims),
        num_sources(num_sources),
        num_terms(num_terms),
        domain_mins(domain_mins, domain_mins+num_dims),
        domain_maxs(domain_maxs, domain_maxs+num_dims),
        do_poisson_solve(do_poisson_solve),
        has_analytic_soln(has_analytic_soln)
  // clang-format on
  {
    assert(num_dims > 0);
    assert(num_sources >= 0);
    assert(num_terms > 0);
  }

  // getters for PDE functions provided by the derived implementation
  virtual std::vector<vector_func<P>> initial_condition_funcs() const = 0;
  virtual std::vector<std::vector<vector_func<P>>>
  source_vector_funcs() const                                     = 0;
  virtual std::vector<scalar_func<P>> source_scalar_funcs() const = 0;
  virtual std::vector<vector_func<P>> exact_vector_funcs() const
  {
    if (has_analytic_soln)
      std::cout << "exact functions expected, but not provided. exiting."
                << '\n';
    else
      std::cout << "PDE has not analytic solutions. exiting." << '\n';

    exit(1);
  }
  virtual scalar_func<P> exact_scalar_func() const
  {
    if (has_analytic_soln)
      std::cout << "exact functions expected, but not provided. exiting."
                << '\n';
    else
      std::cout << "PDE has not analytic solutions. exiting." << '\n';

    exit(1);
  }

  // public but const data. no getters
  int const num_dims;
  int const num_sources;
  int const num_terms;
  std::vector<P> const domain_mins;
  std::vector<P> const domain_maxs;
  bool const do_poisson_solve;
  bool const has_analytic_soln;

  virtual ~PDE() {}

protected:
  void verify() const
  {
    assert(num_dims > 0);
    assert(num_sources >= 0);

    assert(domain_mins.size() == static_cast<unsigned>(num_dims));
    assert(domain_maxs.size() == static_cast<unsigned>(num_dims));

    assert(initial_condition_funcs().size() == static_cast<unsigned>(num_dims));

    if (has_analytic_soln)
    {
      assert(exact_vector_funcs().size() == static_cast<unsigned>(num_dims));
      assert(typeid(exact_scalar_func()) == typeid(scalar_func<P>));
    }

    for (auto const &funcs : source_vector_funcs())
    {
      assert(funcs.size() == static_cast<unsigned>(num_dims));
    }
    assert(source_scalar_funcs().size() == static_cast<unsigned>(num_sources));
  }
};

// ---------------------------------------------------------------------------
//
// the "Vlasov 7" pde
//
// This is the case corresponding to Figure 7 from FIXME
//
// It ... (FIXME explain properties like having an analytic solution, etc.)
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_vlasov7 : public PDE<P>
{
public:
  PDE_vlasov7()
      : PDE<P>(_num_dims, _num_sources, _num_terms, _domain_mins, _domain_maxs,
               _do_poisson_solve, _has_analytic_soln)
  {
    PDE<P>::verify();
  }

  //
  // Specify initial condition vector functions...
  //
  static std::vector<P> initial_condition_x0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(elem * (_domain_maxs[0] - elem));
    }
    return fx;
  }
  static std::vector<P> initial_condition_v0(std::vector<P> const x)
  {
    return std::vector<P>(x.size(), static_cast<P>(0.0));
  }

  //
  // Specify exact solution vectors/time function...
  //
  static std::vector<P> exact_solution_x0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(elem * (static_cast<P>(1.0) - elem));
    }
    return fx;
  }
  static std::vector<P> exact_solution_v0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back((elem - _domain_maxs[1]) * (elem + _domain_maxs[1]));
    }
    return fx;
  }
  static P exact_time(P const time) { return time; }

  //
  // Specify source functions...
  //

  // Source 0
  static std::vector<P> source_0_x0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(elem * (static_cast<P>(1.0) - elem));
    }
    return fx;
  }
  static std::vector<P> source_0_v0(std::vector<P> const x)
  {
    std::vector<P> fx;
    P v_max = _domain_maxs[1];
    for (P const &elem : x)
    {
      fx.push_back((elem - v_max) * (elem + v_max));
    }
    return fx;
  }
  static P source_0_time(P const time)
  {
    return static_cast<P>(1.0) + 0 * time;
  }

  // Source 1
  static std::vector<P> source_1_x0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(static_cast<P>(1.0) - static_cast<P>(2.0) * elem);
    }
    return fx;
  }
  static std::vector<P> source_1_v0(std::vector<P> const x)
  {
    std::vector<P> fx;
    P v_max = _domain_maxs[1];
    for (P const &elem : x)
    {
      fx.push_back(elem * (elem - v_max) * (elem + v_max));
    }
    return fx;
  }
  static P source_1_time(P const time) { return time; }

  // Source 2
  static std::vector<P> source_2_x0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(elem * (static_cast<P>(1.0) - elem));
    }
    return fx;
  }
  static std::vector<P> source_2_v0(std::vector<P> const x)
  {
    std::vector<P> fx;
    for (P const &elem : x)
    {
      fx.push_back(static_cast<P>(2.0) * elem);
    }
    return fx;
  }
  static P source_2_time(P const time) { return time; }

  //
  // Specify electric field...
  //
  // FIXME don't know how we're handling applying a spec. electric field yet
  // std::function<T(T, PARAMS<T>)> exactEx = [&](T x, PARAMS<T> p) {
  // return 1.0; };
  /*   std::vector<T> makeExactEx (std::vector<T> const & x) const override {
         return std::vector<T>(x.size(), 1.0);
     }*/

  // std::function<T(T, PARAMS<T>)> exactEt = [&](T t, PARAMS<T> p) {
  // return 1.0; }; T getExactEt(T t) const override {
  //    return 1.0;
  //}

  //
  // Implement the virtual accessors expected by the PDE<P> base
  //
  std::vector<vector_func<P>> initial_condition_funcs() const override
  {
    return _initial_condition_funcs;
  }
  std::vector<std::vector<vector_func<P>>> source_vector_funcs() const override
  {
    return _source_vector_funcs;
  }
  std::vector<scalar_func<P>> source_scalar_funcs() const override
  {
    return _source_scalar_funcs;
  }
  std::vector<vector_func<P>> exact_vector_funcs() const override
  {
    return _exact_vector_funcs;
  }
  scalar_func<P> exact_scalar_func() const override
  {
    return _exact_scalar_func;
  }

private:
  static int constexpr _num_dims             = 2;
  static int constexpr _num_sources          = 3;
  static int constexpr _num_terms            = 2;
  static bool constexpr _do_poisson_solve    = false;
  static bool constexpr _has_analytic_soln   = true;
  static P constexpr _domain_mins[_num_dims] = {0.0, -5.0};
  static P constexpr _domain_maxs[_num_dims] = {1.0, 5.0};

  //
  // containers holding the function handles to the PDE functions
  //
  std::vector<vector_func<P>> _initial_condition_funcs = {
      &PDE_vlasov7::initial_condition_x0, &PDE_vlasov7::initial_condition_v0};

  std::vector<std::vector<vector_func<P>>> const _source_vector_funcs = {
      {&PDE_vlasov7::source_0_x0, &PDE_vlasov7::source_0_v0},
      {&PDE_vlasov7::source_1_x0, &PDE_vlasov7::source_1_v0},
      {&PDE_vlasov7::source_2_x0, &PDE_vlasov7::source_2_v0}};

  std::vector<scalar_func<P>> const _source_scalar_funcs = {
      &PDE_vlasov7::source_0_time, &PDE_vlasov7::source_1_time,
      &PDE_vlasov7::source_2_time};

  std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_vlasov7::exact_solution_x0, &PDE_vlasov7::exact_solution_v0};

  scalar_func<P> const _exact_scalar_func = &PDE_vlasov7::exact_time;
};

// ---------------------------------------------------------------------------
//
// a user-defined PDE
//
// It requires ... FIXME
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_user : public PDE<P>
{
public:
  PDE_user()
      : PDE<P>(_num_dims, _num_sources, _num_terms, _domain_mins, _domain_maxs,
               _do_poisson_solve, _has_analytic_soln)
  {
    PDE<P>::verify();
  }

  //
  // Specify initial condition vector functions...
  //
  static std::vector<P> initial_condition_x0(std::vector<P> const x)
  {
    return x;
  }
  static std::vector<P> initial_condition_v0(std::vector<P> const x)
  {
    return x;
  }

  //
  // Specify exact solution vectors/time function...
  //
  static std::vector<P> exact_solution_x0(std::vector<P> const x) { return x; }
  static std::vector<P> exact_solution_v0(std::vector<P> const x) { return x; }
  static P exact_time(P const time) { return static_cast<P>(0.0 * time); }

  //
  // Specify source functions...
  //

  // Source 0
  static std::vector<P> source_0_x0(std::vector<P> const x) { return x; }
  static std::vector<P> source_0_v0(std::vector<P> const x) { return x; }
  static P source_0_time(P const time) { return static_cast<P>(0.0 * time); }

  // Source 1
  static std::vector<P> source_1_x0(std::vector<P> const x) { return x; }
  static std::vector<P> source_1_v0(std::vector<P> const x) { return x; }
  static P source_1_time(P const time) { return time; }

  // Source 2
  static std::vector<P> source_2_x0(std::vector<P> const x) { return x; }
  static std::vector<P> source_2_v0(std::vector<P> const x) { return x; }
  static P source_2_time(P const time) { return static_cast<P>(0.0 * time); }

  //
  // Specify electric field...
  //
  // FIXME don't know how we're handling applying a spec. electric field yet
  // std::function<T(T, PARAMS<T>)> exactEx = [&](T x, PARAMS<T> p) {
  // return 1.0; };
  /*   std::vector<T> makeExactEx (std::vector<T> const & x) const override {
         return std::vector<T>(x.size(), 1.0);
     }*/

  // std::function<T(T, PARAMS<T>)> exactEt = [&](T t, PARAMS<T> p) {
  // return 1.0; }; T getExactEt(T t) const override {
  //    return 1.0;
  //}

  //
  // Implement the virtual accessors expected by the PDE<P> base
  //
  std::vector<vector_func<P>> initial_condition_funcs() const override
  {
    return _initial_condition_funcs;
  }
  std::vector<std::vector<vector_func<P>>> source_vector_funcs() const override
  {
    return _source_vector_funcs;
  }
  std::vector<scalar_func<P>> source_scalar_funcs() const override
  {
    return _source_scalar_funcs;
  }
  std::vector<vector_func<P>> exact_vector_funcs() const override
  {
    return _exact_vector_funcs;
  }
  scalar_func<P> exact_scalar_func() const override
  {
    return _exact_scalar_func;
  }

private:
  static int constexpr _num_dims             = 2;
  static int constexpr _num_sources          = 0;
  static int constexpr _num_terms            = 0;
  static bool constexpr _do_poisson_solve    = false;
  static bool constexpr _has_analytic_soln   = false;
  static P constexpr _domain_mins[_num_dims] = {0.0, 0.0};
  static P constexpr _domain_maxs[_num_dims] = {0.0, 0.0};

  //
  // containers holding the function handles to the PDE functions
  //
  std::vector<vector_func<P>> _initial_condition_funcs = {
      &PDE_user::initial_condition_x0, &PDE_user::initial_condition_v0};

  std::vector<std::vector<vector_func<P>>> const _source_vector_funcs = {
      {&PDE_user::source_0_x0, &PDE_user::source_0_v0},
      {&PDE_user::source_1_x0, &PDE_user::source_1_v0},
      {&PDE_user::source_2_x0, &PDE_user::source_2_v0}};

  std::vector<scalar_func<P>> const _source_scalar_funcs = {
      &PDE_user::source_0_time, &PDE_user::source_1_time,
      &PDE_user::source_2_time};

  std::vector<vector_func<P>> const _exact_vector_funcs = {
      &PDE_user::exact_solution_x0, &PDE_user::exact_solution_v0};

  scalar_func<P> const _exact_scalar_func = &PDE_user::exact_time;
};

// ---------------------------------------------------------------------------
//
// A free function factory for making pdes. eventually will want to change the
// return for some of these once we implement them...
//
// ---------------------------------------------------------------------------
template<typename P>
std::unique_ptr<PDE<P>> make_PDE(PDE_opts choice)
{
  switch (choice)
  {
  case PDE_opts::vlasov4:
    return std::make_unique<PDE_vlasov7<P>>(); // FIXME
  case PDE_opts::vlasov43:
    return std::make_unique<PDE_vlasov7<P>>(); // FIXME
  case PDE_opts::vlasov5:
    return std::make_unique<PDE_vlasov7<P>>(); // FIXME
  case PDE_opts::vlasov7:
    return std::make_unique<PDE_vlasov7<P>>();
  case PDE_opts::vlasov8:
    return std::make_unique<PDE_vlasov7<P>>(); // FIXME
  case PDE_opts::pde_user:
    return std::make_unique<PDE_user<P>>();
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}
