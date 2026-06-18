#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file elliptic_nonsep.cpp
 * \brief Non-separable Elliptic equation
 * \author The ASGarD Team
 *
 * Steady-state partial differential equation with non-separable coefficient and boundary conditions.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_ellipticns Example: Non-separable elliptic equation
 *
 * \par Non-separable elliptic equation
 * Solves the 3D elliptic equation
 * \f[ -\nabla \cdot \eta(x, y, z) \nabla f = s(x, y, z) \f]
 * over the domain (0, 1)^3 and the non-separable coefficient is
 * \f[ \eta(x, y, z) = 1 + 0.5 \sin(2 \pi (x + y + z)) \f]
 * and the source is chosen to make the exact solution
 * \f[ f(x, y, z) = \cos(x + y + 2 z) \f]
 * The solution can be obtained by assigning inhomogeneous non-separable boundary conditions,
 * Dirichlet in x and Neumann in y and z.
 *
 * \par
 * This examples shows how to set different types of non-separable coefficients
 * and boundary conditions.
 */

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief Make an elliptic PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme description
 *
 * \snippet elliptic_nonsep.cpp ellipticns make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_elliptic_pde(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns make]
#endif
  options.title = "Non-separable elliptic PDE";

  asgard::pde_domain<P> domain({{0, 1}, {0, 1}, {0, 1}});

  options.default_degree = 2;
  options.default_start_levels = {4, };

  // previous examples were setting a default stepping method
  // which allows the cli options to overwrite the selection
  // here, we are overwriting the cli selection, if another
  // method was requested then a warning will be generated
  // (this should probably be an error instead of a warning)
  options.force_step_method(asgard::time_method::steady);

  // OK for small problems, larger one should switch to gmres or bicgstab
  options.default_solver = asgard::solver_method::gmres;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_inner_iterations = 50;
  options.default_isolver_iterations = 500;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  asgard::term_1d<P> I = asgard::term_identity{};

  asgard::term_md<P> divx = { asgard::term_div<P>{-1}, I, I};
  asgard::term_md<P> divy = { I, asgard::term_div<P>{-1, asgard::boundary_type::bothsides}, I};
  asgard::term_md<P> divz = { I, I, asgard::term_div<P>{-1, asgard::boundary_type::bothsides}};

  asgard::term_md<P> gradx = { asgard::term_grad<P>{1, asgard::boundary_type::bothsides}, I, I};
  asgard::term_md<P> grady = { I, asgard::term_grad<P>{1}, I};
  asgard::term_md<P> gradz = { I, I, asgard::term_grad<P>{1}};

  auto eta = [](P x, P y, P z) -> P { return (1 + P{0.5} * std::sin(P{2 * PI} * (x + y + z))); };

  auto exact = [](P x, P y, P z) -> P { return std::cos(x + y + 2 * z); };

  auto coeff = [=](P, asgard::vector2d<P> const &nodes,
                 std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      // ignore the first input, it is time but it is not implemented yet
      for (size_t i = 0; i < f.size(); i++) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        P const z = nodes[i][2];

        vals[i] = f[i] * eta(x, y, z);
      }
    };

  auto fx0 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of the solution at x = 0, the nodes are at the wall corresponding to x = 0
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        // nodes for the full domain are ordered as (x, y, z), but here
        // we are setting boundary condition in x, which means that
        // we are replacing variable x with a fixed 0, so the nodes
        // are ordered as (y, z)
        P const y = nodes[i][0];
        P const z = nodes[i][1];

        f[i] = exact(0, y, z);
      }
    };

  auto fx1 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of the solution at x = 1
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        P const y = nodes[i][0];
        P const z = nodes[i][1];

        f[i] = exact(1, y, z);
      }
    };

  auto eta_dfy0 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of df/dy at y = 0, the nodes are at the wall corresponding to y = 0
      // must also multiply by the coefficient eta
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        // nodes for the full domain are ordered as (x, y, z)
        // removing y leaves us as (x, z)
        P const x = nodes[i][0];
        P const z = nodes[i][1];;

        f[i] = -eta(x, 0, z) * std::sin(x + 2 * z);
      }
    };

  auto eta_dfy1 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of df/dy at y = 1, the nodes are at the wall corresponding to y = 1
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        P const x = nodes[i][0];
        P const z = nodes[i][1];

        f[i] = -eta(x, 1, z) * std::sin(x + 1 + 2 * z);
      }
    };

  auto eta_dfz0 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of df/dz at z = 0, the nodes are at the wall corresponding to z = 0
      // must also multiply by the coefficient eta
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        // nodes for the full domain are ordered as (x, y, z)
        // removing y leaves us as (x, y)
        P const x = nodes[i][0];
        P const y = nodes[i][1];

        f[i] = -eta(x, y, 0) * std::sin(x + y);
      }
    };

  auto eta_dfz1 = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &f) ->
    void {
      // value of df/dz at z = 1, the nodes are at the wall corresponding to z = 1
      assert(nodes.stride() == 2);
      for (int64_t i = 0; i < nodes.num_strips(); i++) {
        P const x = nodes[i][0];
        P const y = nodes[i][1];

        f[i] = -eta(x, y, 1) * std::sin(x + y + 2);
      }
    };

  gradx += asgard::left_boundary_flux<P>(fx0);
  gradx += asgard::right_boundary_flux<P>(fx1);

  divy += asgard::left_boundary_flux<P>(eta_dfy0);
  divy += asgard::right_boundary_flux<P>(eta_dfy1);

  divz += asgard::left_boundary_flux<P>(eta_dfz0);
  divz += asgard::right_boundary_flux<P>(eta_dfz1);

  asgard::term_md<P> dxx = {divx, asgard::term_interp<P>{coeff}, gradx};
  asgard::term_md<P> dyy = {divy, asgard::term_interp<P>{coeff}, grady};
  asgard::term_md<P> dzz = {divz, asgard::term_interp<P>{coeff}, gradz};

  pde += dxx;
  // pde += dyy;
  // pde += dzz;

  P const dx = pde.cell_size(asgard::dimension_id{0});
  P const dy = pde.cell_size(asgard::dimension_id{1});
  P const dz = pde.cell_size(asgard::dimension_id{2});

  asgard::term_md<P> penx = { asgard::term_penalty<P>{P{1} / dx}, I, I };
  asgard::term_md<P> peny = { I, asgard::term_penalty<P>{P{1} / dy}, I };
  asgard::term_md<P> penz = { I, I, asgard::term_penalty<P>{P{1} / dz} };

  pde += penx;
  // pde += peny;
  // pde += penz;

  auto source = [=](P, asgard::vector2d<P> const &nodes, std::vector<P> &s) ->
    void {
      for (int64_t i = 0; i < nodes.num_strips(); i++)
      {
        P const x = nodes[i][0];
        P const y = nodes[i][1];
        P const z = nodes[i][2];

        P const e   = 1 + 0.5 * std::sin(2 * PI * (x + y + z));
        P const de  = PI * std::cos(2 * PI * (x + y + z));
        P const dde = -2 * PI * PI * std::sin(2 * PI * (x + y + z));

        P const f   =  std::cos(x + y + 2 * z);
        P const df  = -std::sin(x + y + 2 * z); // the z component is multiplied by 2
        P const ddf = -std::cos(x + y + 2 * z); // the z component is multiplied by 4

        std::ignore = f;
        std::ignore = dde;

        // s[i] = -(6 * e * ddf + 4 * de * df); // using all 3 derivative components in x y z

        // using only derivative in x
        s[i] = -(e * ddf + de * df);

        // s[i] = 0; // disable the source term, sources will come only from the boundary
      }
    };

  pde += asgard::source<P>(source);

  // if an initial condition is specified, it will be used as the initial guess
  // of an iterative solver, otherwise zeros is used as the initial guess

  return pde;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns make]
#endif
}

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_elliptic_pde() and the solution should be set.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet elliptic_nonsep.cpp ellipticns get-err
 */
template<typename P>
double get_error_max(asgard::discretization_manager<P> const &disc)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns get-err]
#endif

  int const np = 20;

  // makes a dense grid over the domain using np points each direction
  asgard::vector2d<double> const mesh = asgard::make_grid<double>(disc.domain(), np);

  std::vector<double> ref(mesh.num_strips());
  std::vector<double> con(mesh.num_strips());

  #pragma omp parallel for
  for (int64_t i = 0; i < mesh.num_strips(); i++)
    ref[i] = std::cos(mesh[i][0] + mesh[i][1] + 2 * mesh[i][2]);

  auto shot = disc.get_snapshot_mpi();

  shot.reconstruct(mesh[0], mesh.num_strips(), con.data());

  double err = 0;
  double nrm = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    err = std::max(err, std::abs(con[i] - ref[i]));
    nrm = std::max(nrm, std::abs(ref[i]));
  }

  return err / nrm;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [ellipticns get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_ellipticns
 * \brief main() for the sine-wave example
 *
 * The main() processes the command line arguments and calls both
 * make_elliptic_pde() and get_error_l2().
 *
 * \snippet elliptic_nonsep.cpp elliptic main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic main]
#endif
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
-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  #ifdef ASGARD_USE_GPU
  std::cerr << "Interpolated boundary conditions not available for the GPU ... yet.\n";
  return 0;
  #endif

  auto pde = make_elliptic_pde(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_max(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- steady state error: " << err << '\n';

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic main]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, one should only include what is needed
using namespace asgard;

template<typename P>
void dotest(double tol, int num_dims, std::string const &opts) {
  current_test<P> test_(opts, num_dims);

  auto options = make_opts(opts);

  auto pde = make_elliptic_pde<P>(options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_max(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("elliptic steady state problem", " div.grad f = sources");

  std::cerr << "  EXAMPLE INCOMPLETE, TESTS ARE NOT WORKING YET\n";

  #ifdef ASGARD_ENABLE_DOUBLE
  // dotest<double>(5.E-3, 1, "-d 1 -l 3");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  // dotest<float>(5.E-3, 1, "-d 1 -l 5");
  // dotest<float>(5.E-3, 1, "-d 2 -l 3");
  // dotest<float>(5.E-3, 1, "-d 2 -l 3 -bc 1");
  #endif
}

#endif
