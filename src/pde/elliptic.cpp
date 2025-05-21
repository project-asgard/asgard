#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file elliptic.cpp
 * \brief Elliptic equation
 * \author The ASGarD Team
 *
 * Simple example of steady-state partial differential equation.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_elliptic Example: Elliptic equation
 *
 * \par Elliptic equation
 * Creates a simple elliptic PDE that multiplies across the dimensions
 * the same one-dimensional boundary value problem
 * \f[ \frac{d^2}{d x^2} f = 2 \f]
 * the domain is (0, 1) and the exact solution is
 * \f[ f(x) = 2 x - x^2 \f]
 * The solution can be obtained by assigning homogeneous boundary conditions,
 * Dirichlet on the left and Neumann on the right,
 * or alternatively we can assign inhomogeneous conditions
 * \f[ \frac{d}{dx} f(0) = 2, \qquad f(1) = 1 \f]
 * Since the solution is a quadratic function, using degree of 2 or more
 * should resolve the exact solution regardless of the grid
 * (up to rounding error due to conditioning and precision).
 *
 * \par
 * This examples shows how to set different types of boundary conditions
 * and how to solve a steady state problem.
 */

/*!
 * \ingroup asgard_examples_elliptic
 * \brief Indicates the type of boundary conditions to use
 */
enum class boundary_enum {
  //! Dirichlet and Neumann set to zero value
  homogeneous,
  //! Dirichlet and Neumann set to non-zero value
  inhomogeneous
};

/*!
 * \ingroup asgard_examples_elliptic
 * \brief Make an elliptic PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam boudnary indicates the type of boundary to use
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims number of dimensions
 * \param options is the set of options
 *
 * \returns the PDE description, the \b v2 suffix is temporary syntax and will be
 *          removed in the near future
 *
 * \b Note: The asgard namespace includes the name \b boundary_type,
 * it a natural name but it is possible to create a conflict if the entire namespace
 * is included.
 *
 * \snippet elliptic.cpp elliptic make
 */
template<boundary_enum boundary, typename P = asgard::default_precision>
asgard::pde_scheme<P> make_elliptic_pde(int num_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic make]
#endif
  rassert(1 <= num_dims and num_dims <= 6, "invalid number of dimensions");

  options.title = "Elliptic PDE " + std::to_string(num_dims) + "D";

  asgard::pde_domain<P> domain(std::vector<asgard::domain_range>(num_dims, {0, 1}));

  options.default_degree = 1;
  options.default_start_levels = {4, };

  // previous examples were setting a default stepping method
  // which allows the cli options to overwrite the selection
  // here, we are overwriting the cli selection, if another
  // method was requested then a warning will be generated
  // (this should probably be an error instead of a warning)
  options.force_step_method(asgard::time_method::steady);

  // OK for small problems, larger one should switch to gmres or bicgstab
  options.default_solver = asgard::solver_method::direct;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  // s1d is the exact solution in 1d
  auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = x[i] * (P{2} - x[i]);
    };

  // "exact" is the solution in multiple dimensions
  asgard::separable_func<P> exact(std::vector<asgard::svector_func1d<P>>(num_dims, s1d),
                                  asgard::ignores_time);

  if constexpr (boundary == boundary_enum::homogeneous)
  {
    // fixed boundary set to the div term corresponds to Neumann boundary
    asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
                                                 asgard::boundary_type::right);
    // fixed boundary set to the grad term corresponds to Dirichlet boundary
    asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
                                                   asgard::boundary_type::left);

    // the multi-dimensional operator, initially set to identity in md
    std::vector<asgard::term_1d<P>> ops(num_dims);
    for (int d = 0; d < num_dims; d++)
    {
      // combine the div and grad into a single chain term
      asgard::term_1d<P> fxx({div, grad});

      // based on the domain and max-level, get the cell-size in direction d
      P const dx = pde.cell_size(d);

      // adding penalty to stabilize the steady state equation
      // the penalty is applied only to discontinuities, if the solution is continuous
      // then the penalty will not alter the result, this only improves the conditioning
      fxx.set_penalty(P{1} / dx);

      // add the second order operator in dimension dim
      ops[d] = fxx;
      pde += asgard::term_md<P>(ops);
      ops[d] = asgard::term_identity{};
    }

  } else { // inhomogeneous case

    // allowing for inhomogeneous boundary, we can use many combinations of
    // Dirichlet and Neumann data
    // the 1D case is set for Dirichlet boundary
    // the mD case is set for mix Dirichlet and Neumann conditions

    if (num_dims == 1)
    {
      // fixed boundary set to the div term corresponds to Neumann boundary
      asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind);

      // fixed boundary set to the grad term corresponds to Dirichlet boundary
      asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
                                                     asgard::boundary_type::bothsides);
      // merge the div and grad terms
      asgard::term_1d<P> fxx({div, grad});

      // penalize discontinuities
      P const dx = pde.min_cell_size();
      fxx.set_penalty(P{1} / dx);

      // merge into a multi-dimensional term with one dimension
      asgard::term_md<P> fxx_md({fxx, });

      // adding inhomogeneous term to the right of the domain
      // starting with the exact solution
      asgard::separable_func<P> bc = exact;
      // the 0-th dimension component is set to constant 1
      bc.set(0, P{1});
      // add the condition at the right point
      fxx_md += asgard::right_boundary_flux(bc);

      // add the term with inhomogeneous boundary to the pde
      pde += fxx_md;
    }
    else
    {
      // setting fixed boundary for the div term in the chain
      // results in Neumann conditions imposed on the field
      // think of this as imposing Dirichlet condition on the output of the grad term
      // and the output of the grad term is the derivative of the field
      asgard::term_1d<P> div = asgard::term_div<P>(-1, asgard::flux_type::upwind,
                                                   asgard::boundary_type::left);

      // Dirichlet boundary set to the grad term corresponds to Dirichlet boundary
      asgard::term_1d<P> grad = asgard::term_grad<P>(1, asgard::flux_type::upwind,
                                                     asgard::boundary_type::right);
      // merge the div and grad terms
      asgard::term_1d<P> fxx({div, grad});

      // penalize discontinuities
      P const dx = pde.min_cell_size();
      fxx.set_penalty(P{1} / dx);

      for (int d = 0; d < num_dims; d++)
      {
        // make vector of terms_1d for each dimension
        std::vector<asgard::term_1d<P>> terms(num_dims);
        terms[d] = fxx;

        // merge into a multi-dimensional term with one dimension
        asgard::term_md<P> fxx_md(terms);

        // setting Dirichlet condition 1 on the right wall of dimension d
        // by default, the boundary condition is applied to the field
        // that is the input of the term, i.e., the input to the grad term
        asgard::separable_func<P> bc = exact;
        bc.set(d, P{1});
        fxx_md += asgard::right_boundary_flux(bc);

        // setting Neumann condition 2 on the left wall of dimension d
        bc = exact;
        bc.set(d, P{2});
        asgard::boundary_flux<P> lbf = asgard::left_boundary_flux(bc);
        // at this point we have the boundary flux
        // but we also need to apply it to the input of the div-term,
        // i.e., set the level of the chain to the index of the div term
        lbf.chain_level(d) = 0;
        fxx_md += lbf;

        // add the term with boundary conditions to the pde
        pde += fxx_md;
      }
    }
  }

  for (int d = 0; d < num_dims; d++) {
    // using separability properties, copy over the exact solution
    asgard::separable_func<P> src = exact;
    // differentiate in the d-th direction, i.e., replace the function
    // with a constant 2
    src.set(d, 2);

    pde.add_source(std::move(src));
  }

  // if an initial condition is specified, it will be used as the initial guess
  // of an iterative solver, other zeros is used as the initial guess
  // the direct solver does not use an initial guess

  return pde;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic make]
#endif
}

/*!
 * \ingroup asgard_examples_elliptic
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
 * \snippet elliptic.cpp elliptic get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic get-err]
#endif

  int const num_dims = disc.num_dims();

  // see the continuity example for the orthogonality trick

  // construct the exact solution, since there is no initial condition
  auto s1d = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = x[i] * (P{2} - x[i]);
    };

  // set the right-hand-side for each dimension
  std::vector<asgard::svector_func1d<P>> func(num_dims, s1d);

  std::vector<P> const eref = disc.project_function(asgard::separable_func<P>(func));

  double constexpr space1d = 8.0 / 15.0; // integral of (2x - x^2)^2 over (0, 1)

  // this is the L^2 norm-squared of the exact solution
  double const enorm = asgard::fm::powi(space1d, num_dims);

  disc.sync_mpi_state(); // is using multiple ranks, sync across the ranks
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

#ifndef __ASGARD_DOXYGEN_SKIP
//! [elliptic get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

/*!
 * \ingroup asgard_examples_elliptic
 * \brief main() for the sine-wave example
 *
 * The main() processes the command line arguments and calls both
 * make_elliptic_pde() and get_error_l2().
 *
 * \snippet elliptic.cpp elliptic main
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
    std::cout << "\n solves an elliptic PDE Laplacian f = 2 * num-dims:\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-dims            -dm     int        accepts: 1 - 6
                                    the number of dimensions
-bound           -bc     int        accepts: 0 or 1
                                    0 - use homogeneous boundary conditions
                                    1 - use inhomogeneous boundary conditions

-test                               perform self-testing
)help";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {"-dims", "-dm", "-bound", "-bc"});

  if (options.has_cli_entry("-test")) {
    self_test();
    return 0;
  }

  // using a method similar to extra_cli_value that will loop for multiple entries
  // returns the first entry found
  std::optional<int> cli_btype = options.extra_cli_value_group<int>({"-bound", "-bc"});
  int const btype = cli_btype.value_or(0);

  if (options.is_mpi_rank_zero()) {
    if (not cli_btype) {
      std::cout << "using default homogeneous boundary\n";
    } else if (btype == 0) {
      std::cout << "using homogeneous boundary\n";
    } else if (btype == 1) {
      std::cout << "using inhomogeneous boundary\n";
    } else {
      std::cerr << "incorrect value for -bound, must use 0 or 1\n";
      return 1;
    }
  }

  // setting the dimensions
  std::optional<int> cli_dims = options.extra_cli_value_group<int>({"-dims", "-dm"});
  int const num_dims = cli_dims.value_or(2);

  if (options.is_mpi_rank_zero()) {
    if (not cli_dims) {
      std::cout << "setting default 2D problem\n";
    } else {
      std::cout << "setting " << num_dims << "D problem\n";
    }
  }

  auto pde = (btype == 0)
             ? make_elliptic_pde<boundary_enum::homogeneous, P>(num_dims, options)
             : make_elliptic_pde<boundary_enum::inhomogeneous, P>(num_dims, options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::low);

  disc.advance_time();

  disc.final_output();

  P const err = get_error_l2(disc);
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

  // using a method similar to extra_cli_value that will loop for multiple entries
  // returns the first entry found
  int const btype = options.extra_cli_value<int>("-bc").value_or(0);

  auto pde = (btype == 0)
             ? make_elliptic_pde<boundary_enum::homogeneous, P>(num_dims, options)
             : make_elliptic_pde<boundary_enum::inhomogeneous, P>(num_dims, options);

  asgard::discretization_manager<P> disc(std::move(pde), asgard::verbosity_level::quiet);

  disc.advance_time();

  double const err = get_error_l2(disc);
  // std::cout << err << '\n';
  tcheckless(1, err, tol);
}

void self_test() {
  all_tests testing_("elliptic steady state problem", " div.grad f = sources");

  #ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(5.E-3, 1, "-d 1 -l 3");
  dotest<double>(1.E-3, 1, "-d 1 -l 4");
  dotest<double>(5.E-4, 1, "-d 1 -l 5");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");
  dotest<double>(5.E-4, 1, "-d 1 -l 5 -bc 1");

  dotest<double>(1.E-3, 2, "-d 1 -l 4");
  dotest<double>(1.E-3, 3, "-d 1 -l 5 -sv bicgstab");
  dotest<double>(1.E-3, 4, "-d 1 -l 5 -sv bicgstab");

  dotest<double>(1.E-7, 1, "-d 2 -l 3");
  dotest<double>(5.E-7, 2, "-d 2 -l 3");
  dotest<double>(5.E-7, 3, "-d 2 -l 3");

  dotest<double>(1.E-3, 1, "-d 1 -l 4");
  dotest<double>(1.E-3, 2, "-d 1 -l 5");
  dotest<double>(1.E-3, 3, "-d 1 -l 6  -sv bicgstab");

  dotest<double>(1.E-7, 1, "-d 2 -l 3 -bc 1");
  dotest<double>(5.E-7, 2, "-d 2 -l 3 -bc 1");
  dotest<double>(5.E-7, 3, "-d 2 -l 3 -bc 1");

  dotest<double>(1.E-3, 1, "-d 1 -l 4 -bc 1");
  dotest<double>(1.E-3, 2, "-d 1 -l 5 -bc 1");
  dotest<double>(1.E-3, 3, "-d 1 -l 6 -bc 1 -sv bicgstab");

  dotest<double>(1.E-3, 2, "-bc 1 -l 3 -m 8 -a 1.E-5");
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  dotest<float>(5.E-3, 1, "-d 1 -l 5");
  dotest<float>(5.E-3, 1, "-d 2 -l 3");
  dotest<float>(5.E-3, 1, "-d 2 -l 3 -bc 1");

  dotest<float>(5.E-3, 2, "-d 1 -l 5");
  #endif
}

#endif
