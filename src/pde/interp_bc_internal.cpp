#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

using namespace asgard;

enum class pde_mode { first, second, first_nonsep, second_nonsep };

template<typename P = asgard::default_precision, pde_mode mode>
asgard::pde_scheme<P> make_3d_pde(asgard::prog_opts options)
{
  options.title = "Non-separable boundary value PDE, 3D";
  if constexpr (mode == pde_mode::first) {
    options.title += " (first order)";
  } else if constexpr (mode == pde_mode::first_nonsep) {
    options.title += " (first order, non-sep)";
  } else if constexpr (mode == pde_mode::second_nonsep) {
    options.title += " (second order, non-sep)";
  } else {
    options.title += " (second order)";
  }

  pde_domain<P> domain({{0, 1}, {0, 1}, {0, 1}});

  options.default_degree = 2;
  options.default_start_levels = {4, };

  options.force_step_method(time_method::steady);

  options.default_solver = solver_method::gmres;

  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_inner_iterations = 50;
  options.default_isolver_iterations = 1000;

  pde_scheme<P> pde(options, std::move(domain));

  term_1d<P> I = term_identity{};

  if constexpr (mode == pde_mode::first or mode == pde_mode::first_nonsep)
  {
    term_md<P> divx = { term_div<P>{-1, boundary_type::right}, I, I };

    auto fx1 = [=](P, vector2d<P> const &nodes, std::vector<P> &f) ->
      void {
        if constexpr (mode == pde_mode::first) {
          std::ignore = nodes;
          std::fill(f.begin(), f.end(), P{1});
        } else {
          for (int64_t i = 0; i < nodes.num_strips(); i++)
            f[i] = P{1} + nodes[i][0] + nodes[i][1];
        }
      };

    divx += right_boundary_flux<P>(fx1);

    pde += divx;

    P const dx = pde.cell_size(dimension_id{0});

    pde += { term_penalty<P>{P{1} / dx}, I, I };

    // source is the same (constant -1), set in 2 ways for stress-testing
    auto src = [&]() -> auto {
        if constexpr (mode == pde_mode::first) {
          // set via separable function
          auto s = separable_func<P>::const_one(number_of_dimensions{3});
          s.set(dimension_id{0}, -1);
          return s;
        } else {
          // set as if it is non-separable
          return [=](P, vector2d<P> const &nodes, std::vector<P> &s) ->
              void {
                std::ignore = nodes;
                std::fill(s.begin(), s.end(), P{-1});
              };
        }
      }();

    pde += source<P>(src);
  }
  else
  {
    term_md<P> divx  = { term_div<P>{-1}, I, I };
    term_md<P> gradx = { term_grad<P>{1, boundary_type::bothsides}, I, I };

    auto fx0 = [=](P, vector2d<P> const &nodes, std::vector<P> &f) ->
      void {
        if constexpr (mode == pde_mode::second) {
          std::ignore = nodes;
          std::fill(f.begin(), f.end(), P{0});
        } else {
          for (int64_t i = 0; i < nodes.num_strips(); i++) {
            f[i] = nodes[i][0] + nodes[i][1];
          }
        }
      };

    auto fx1 = [=](P, vector2d<P> const &nodes, std::vector<P> &f) ->
      void {
        if constexpr (mode == pde_mode::second) {
          std::ignore = nodes;
          std::fill(f.begin(), f.end(), P{1});
        } else {
          for (int64_t i = 0; i < nodes.num_strips(); i++)
            f[i] = P{1} + nodes[i][0] + nodes[i][1];
        }
      };

    if constexpr (mode == pde_mode::second_nonsep) {
      gradx += left_boundary_flux<P>(fx0);
    } else {
      std::ignore = fx0;
    }

    gradx += right_boundary_flux<P>(fx1);

    pde += term_md<P>{divx, gradx};

    P const dx = pde.cell_size(dimension_id{0});

    pde += { term_penalty<P>{P{1} / dx}, I, I };
  }

  return pde;
}

template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_3d_pde(asgard::prog_opts options)
{
  int count = 0;
  for (auto const &s : {std::string("-first"), std::string("-second"),
                        std::string("-nonsep-1"), std::string("-nonsep-2")})
    if (options.has_cli_entry(s))
      count++;
  rassert(count < 2, "cannot use multiple PDE type switches, e.g., -second and -nonsep-1");

  if (options.has_cli_entry("-second"))
    return make_3d_pde<P, pde_mode::second>(options);
  else if (options.has_cli_entry("-nonsep-1"))
    return make_3d_pde<P, pde_mode::first_nonsep>(options);
  else if (options.has_cli_entry("-nonsep-2"))
    return make_3d_pde<P, pde_mode::second_nonsep>(options);
  else
    return make_3d_pde<P, pde_mode::first>(options);
}

template<typename P>
double get_error_max(discretization_manager<P> const &disc)
{
  int const np = 20;

  // makes a dense grid over the domain using np points each direction
  vector2d<double> const mesh = make_grid<double>(disc.domain(), np);

  std::vector<double> ref(mesh.num_strips());
  std::vector<double> con(mesh.num_strips());

  if (disc.title_contains("non-sep")) {
    #pragma omp parallel for
    for (int64_t i = 0; i < mesh.num_strips(); i++)
      ref[i] = mesh[i][0] + mesh[i][1] + mesh[i][2];
  } else {
    #pragma omp parallel for
    for (int64_t i = 0; i < mesh.num_strips(); i++)
      ref[i] = mesh[i][0];
  }

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
-first                              use a first order problem (default)
-second                             use a second order problem
-nonsep-1                           using first order pde with non-separable solution
-nonsep-2                           using second order pde with non-separable solution
-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", "-first", "-second", "-nonsep-1", "-nonsep-2"}, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  discretization_manager<P> disc(make_3d_pde<P>(options), verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_max(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
}

#ifndef __ASGARD_DOXYGEN_SKIP
template<typename P>
void dotest(double tol, std::string const &opts) {
  current_test<P> test_(opts, 3);

  auto options = make_opts(opts);

  asgard::discretization_manager<P> disc(make_3d_pde<P>(options), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_max(disc);
  // std::cout << err << "    " << tol << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("non-separable bc", " using different PDEs");

  #ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(1.E-9, "-d 1 -l 2");
  dotest<double>(1.E-9, "-d 2 -l 2");
  dotest<double>(1.E-9, "-d 2 -l 2 -second");
  dotest<double>(1.E-9, "-d 3 -l 1 -second");

  dotest<double>(1.E-9, "-d 2 -l 3 -nonsep-1");
  dotest<double>(1.E-9, "-d 2 -l 3 -nonsep-2");

  dotest<double>(1.E-9, "-d 2 -l 5 -nonsep-2");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(5.E-6, "-d 1 -l 2");
  dotest<float>(5.E-6, "-d 2 -l 3 -nonsep-1");
  #endif
}

#endif
