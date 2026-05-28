#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

using namespace asgard;

enum class pde_order { first, second };

template<typename P = asgard::default_precision, pde_order pdeorder>
asgard::pde_scheme<P> make_3d_pde(asgard::prog_opts options)
{
  options.title = "Non-separable boundary value PDE, 3D";
  if constexpr (pdeorder == pde_order::first) {
    options.title += " (first order)";
  } else {
    options.title += " (second order)";
  }

  pde_domain<P> domain({{0, 1}, {0, 1}, {0, 1}});

  options.default_degree = 2;
  options.default_start_levels = {4, };

  options.force_step_method(time_method::steady);

  options.default_solver = solver_method::bicgstab;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> I = term_identity{};

  if constexpr (pdeorder == pde_order::first)
  {
    term_md<P> divx = { term_div<P>{-1, boundary_type::right}, I, I };

    auto fx1 = [=](P, vector2d<P> const &, std::vector<P> &f) ->
      void {
        std::fill(f.begin(), f.end(), P{1});
      };

    divx += right_boundary_flux<P>(fx1);

    pde += divx;

    P const dx = pde.cell_size(dimension_id{0});

    term_md<P> pen = { term_penalty<P>{P{1} / dx, boundary_type::right}, I, I };

    pen += right_boundary_flux<P>(fx1);

    pde += pen;

    auto src = [=](P, vector2d<P> const &, std::vector<P> &s) ->
      void {
        std::fill(s.begin(), s.end(), P{-1});
      };

    pde += source<P>(src);
  }
  else
  {
    term_md<P> divx  = { term_div<P>{-1}, I, I };
    term_md<P> gradx = { term_grad<P>{1, boundary_type::bothsides}, I, I };

    auto fx1 = [=](P, vector2d<P> const &, std::vector<P> &f) ->
      void {
        std::fill(f.begin(), f.end(), P{1});
      };

    gradx += right_boundary_flux<P>(fx1);

    pde += term_md<P>{divx, gradx};

    P const dx = pde.cell_size(dimension_id{0});

    term_md<P> pen = { term_penalty<P>{P{1} / dx, boundary_type::bothsides}, I, I };

    pen += right_boundary_flux<P>(fx1);

    pde += pen;

    auto src = [=](P, vector2d<P> const &, std::vector<P> &s) ->
      void {
        std::fill(s.begin(), s.end(), P{0});
      };

    pde += source<P>(src);
  }

  return pde;
}

template<typename P>
double get_error_max(discretization_manager<P> const &disc)
{
  int const np = 20;

  // makes a dense grid over the domain using np points each direction
  vector2d<double> const mesh = make_grid<double>(disc.domain(), np);

  std::vector<double> ref(mesh.num_strips());
  std::vector<double> con(mesh.num_strips());

  #pragma omp parallel for
  for (int64_t i = 0; i < mesh.num_strips(); i++)
    ref[i] = mesh[i][0];

  auto shot = disc.get_snapshot_mpi();

  shot.reconstruct(mesh[0], mesh.num_strips(), con.data());

  double err = 0;
  double nrm = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    err = std::max(err, std::abs(con[i] - ref[i]));
    nrm = std::max(nrm, std::abs(ref[i]));
  }

  return err / nrm;
}

void self_test();

int main(int argc, char** argv)
{
  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves an non-separable elliptic PDE\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-second                             use a second order problem (default is first order)
-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "-second"}, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  pde_scheme<P> pde = (options.has_cli_entry("-second"))
                      ? make_3d_pde<P, pde_order::second>(options)
                      : make_3d_pde<P, pde_order::first>(options);
  discretization_manager<P> disc(std::move(pde), verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_max(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
}

#ifndef __ASGARD_DOXYGEN_SKIP
template<typename P>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  auto pde = make_3d_pde<P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_max(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("elliptic steady state problem", " div.grad f = sources");

  #ifdef ASGARD_ENABLE_DOUBLE
  // dotest<double>(5.E-3, 1, "-d 1 -l 3");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  // dotest<float>(5.E-3, 1, "-d 1 -l 5");
  #endif
}

#endif
