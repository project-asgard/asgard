#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

class tpde final : public PDE<prec>
{
public:
  using P = prec;

  tpde(int num_dims, int levels, int degree,
       std::function<fk::vector<prec>(fk::vector<prec> const &, prec)> f,
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
  int level = 2;
  int degree = 0;

  basis::wavelet_transform<double, resource::host> wav(level, degree);

  std::array<fk::matrix<double>, 4> mats = generate_multi_wavelets<double>(degree);

  std::cout << std::scientific;
  std::cout.precision(16);

  for (auto const &m : mats)
  {
    std::cout << " ------------------------------------- \n";
    for (int r = 0; r < m.nrows(); r++)
    {
      for (int c = 0; c < m.ncols(); c++)
        std::cout << std::setw(25) << m(r, c);
      std::cout << "\n";
    }
  }

  std::cout << " ------------------------------------- \n";
  std::cout << " ------------------------------------- \n";
  //auto const &T = wav.get_blocks().back();

  for (auto const &T : wav.get_blocks())
  {
    std::cout << " ------------------------------------- \n";
    for (int r = 0; r < T.nrows(); r++)
    {
      for (int c = 0; c < T.ncols(); c++)
        std::cout << std::setw(25) << T(r, c);
      std::cout << "\n";
    }
  }

  fk::matrix<double> I(4, 4);
  for (int i = 0; i < I.ncols(); i++) I(i, i) = 1.0;

  auto P = wav.apply(I, level, basis::side::right, basis::transpose::trans);

  std::cout << " ------------------------------------- \n";
  std::cout << " ------------------------------------- \n";
  for (int r = 0; r < P.nrows(); r++)
  {
    for (int c = 0; c < P.ncols(); c++)
      std::cout << std::setw(25) << P(r, c);
    std::cout << "\n";
  }

  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior
  return 0;
}
