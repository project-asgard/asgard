#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

using namespace asgard;

// testing contrived PDEs with MANY chains
// stress-tests the chaining process

template<typename P>
asgard::pde_scheme<P> make_1d_pde_out(asgard::prog_opts options)
{
  options.title = "5 chain link PDE in 1D";
  options.subtitle = "using outer chains";

  pde_domain<P> domain({{0, 2}, });

  options.default_degree = 2;
  options.default_start_levels = {5, };

  options.force_step_method(time_method::steady);

  options.default_solver = solver_method::bicgstab;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 5000;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> vol2  = term_volume<P>{2};
  term_1d<P> vol05 = term_volume<P>{0.5};

  term_1d<P> volc  = term_volume<P>{vectorize<P>([](P x) -> P { return (P{1} + x); }) };

  term_1d<P> div  = term_div<P>{-1, boundary_type::right};
  term_1d<P> grad = term_grad<P>{1, boundary_type::left};

  term_md<P> c0 = {vol2, };
  term_md<P> c1 = {div, };
  term_md<P> c2 = {volc, };
  term_md<P> c3 = {vol05, };
  term_md<P> c4 = {grad, };

  auto lbc = separable_func<P>::const_one(number_of_dimensions{1});
  auto rbc = separable_func<P>::const_one(number_of_dimensions{1});
  // the right condition must include the exact solution AND the coefficients
  rbc.set(dimension_id{0}, - P{4} * std::exp(-P{4}) * P{1.5});

  c4 += left_boundary_flux{lbc};
  c1 += right_boundary_flux{rbc};

  pde += {c0, c1, c2, c3, c4};

  P const dx = pde.cell_size(dimension_id{0});
  pde += term_md<P>{term_penalty<P>{1 / dx}, };

  pde += separable_func<P>({vectorize<P>([](P x) -> P {
      return (2 + 4 * x - 4 * x * x - 4 * x * x * x) * std::exp(- x * x);
    }), });

  return pde;
}

template<typename P>
asgard::pde_scheme<P> make_1d_pde_in(asgard::prog_opts options)
{
  options.title = "5 chain link PDE in 1D";
  options.subtitle = "using inner chains";

  pde_domain<P> domain({{0, 2}, });

  options.default_degree = 2;
  options.default_start_levels = {5, };

  options.force_step_method(time_method::steady);

  options.default_solver = solver_method::bicgstab;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 5000;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> vol2  = term_volume<P>{2};
  term_1d<P> vol05 = term_volume<P>{0.5};

  term_1d<P> volc  = term_volume<P>{vectorize<P>([](P x) -> P { return (P{1} + x); }) };

  term_1d<P> div  = term_div<P>{-1, boundary_type::right};
  term_1d<P> grad = term_grad<P>{1, boundary_type::left};

  term_1d<P> chain = std::vector<term_1d<P>>{vol2, div, volc, vol05, grad};
  term_md<P> trm = {chain, };

  auto lbc = separable_func<P>::const_one(number_of_dimensions{1});
  auto rbc = separable_func<P>::const_one(number_of_dimensions{1});
  // the right condition must include the exact solution AND the coefficients
  rbc.set(dimension_id{0}, - P{4} * std::exp(-P{4}) * P{1.5});

  trm += left_boundary_flux{lbc, {4, }};
  trm += right_boundary_flux{rbc, {1, }};

  pde += trm;

  P const dx = pde.cell_size(dimension_id{0});
  pde += term_md<P>{term_penalty<P>{1 / dx}, };

  pde += separable_func<P>({vectorize<P>([](P x) -> P {
      return (2 + 4 * x - 4 * x * x - 4 * x * x * x) * std::exp(- x * x);
    }), });

  return pde;
}


template<typename P>
double get_error_l2(discretization_manager<P> const &disc)
{
  std::vector<P> eref;
  double enorm = 0;

  if (disc.title_contains("1D"))
  {
    separable_func<P> exact({vectorize<P>([](P x) -> P { return std::exp(- x * x); }), });

    eref = disc.project_function(exact);

    enorm = 0.626617374642614;
  }

  disc.sync_mpi_state();

  std::vector<P> const &state = disc.current_state();
  assert(eref.size() == state.size());

  double nself = 0;
  double ndiff = 0;
  for (size_t i = 0; i < state.size(); i++)
  {
    double const e = eref[i] - state[i];
    ndiff += e * e;
    double const r = eref[i];
    nself += r * r;
  }

  return std::sqrt((ndiff + std::abs(enorm - nself)) / enorm);
}

template<typename P>
void dotest1d(double tol, std::string const &opts) {
  current_test<P> test_(opts, 1);

  auto options = make_opts(opts);

  asgard::discretization_manager<P> disc(make_1d_pde_out<P>(options), asgard::verbosity_level::quiet);
  tassert(disc.subtitle_contains("outer"));

  disc.advance_time();

  double const err1 = get_error_l2(disc);
  // std::cout << err1 << "    " << tol << '\n';
  tcheckless(1, err1, tol);

  // same test but using internal chaining
  disc = asgard::discretization_manager<P>(make_1d_pde_in<P>(options), asgard::verbosity_level::quiet);
  tassert(disc.subtitle_contains("inner")); // check if the PDE was updated

  disc.advance_time();

  double const err2 = get_error_l2(disc);
  // std::cout << err2 << "    " << tol << '\n';
  tcheckless(1, err2, tol);
}

int main(int argc, char** argv)
{
  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  all_tests testing_("long chains", " using different PDEs");


  #ifdef ASGARD_ENABLE_DOUBLE
  dotest1d<double>(1.E-4, "-d 2 -l 4");
  dotest1d<double>(1.E-5, "-d 2 -l 5");
  dotest1d<double>(5.E-5, "-d 3 -l 3");
  dotest1d<double>(1.E-6, "-d 3 -l 4");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  //
  #endif


  return 0;
}
