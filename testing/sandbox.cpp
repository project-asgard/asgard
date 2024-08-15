#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

class tpde final : public PDE<double>
{
public:
  using P = double;

  tpde(int num_dims, int levels, int degree,
       std::function<fk::vector<double>(fk::vector<double> const &, double)> f,
       prog_opts const &options = prog_opts())
  {
    int constexpr num_sources       = 0;
    int constexpr num_terms         = 1;
    bool constexpr do_poisson_solve = false;
    // disable implicit steps in IMEX
    bool constexpr do_collision_operator = false;
    bool constexpr has_analytic_soln     = false;



    std::vector<dimension<P>> dims(
        num_dims, dimension<P>(-1.0, 0.0, levels, degree,
                               f, nullptr, "x"));

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



int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
