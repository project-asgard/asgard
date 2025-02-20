#include "asgard.hpp"

/*!
 * \internal
 * \file continuity_2d.cpp
 * \brief Simple continuity example
 * \author The ASGarD Team
 * \ingroup asgard_examples_continuity_2d
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_continuity_2d Example 1, 2D continuity equation
 *
 * \par Example 1
 * Solves the continuity partial differential equation
 * \f[ \frac{\partial}{\partial t} f + \nabla \cdot f = s \f]
 * where both \b f and \b s are defined over the two dimensional domain
 * \f[ (x, y) \in (-1, 1) \otimes (-2, 2) \f]
 *
 * \par
 * The right right-hand-side source is chosen so the exact solution
 * is a simple separable function
 * \f[ f(t, x, y) = \sin(2 t) \cos(\pi x) \sin(2 \pi y) \f]
 *
 * \par
 * The example comes with a companion file continuity_2d.py that demonstrates
 * plotting with Python and matplotlib.
 */

/*!
 * \ingroup asgard_examples_continuity_2d
 * \brief Default precision for this example, favors double-precision
 *
 * if ASGarD is compiled with double precision, this defaults to double
 * if only single precision is avaiable, this will be float
 */
using precision = asgard::default_precision;

/*!
 * \ingroup asgard_examples_continuity_2d
 * \brief The ratio of circumference to diameter of a circle
 */
precision constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_examples_continuity_2d
 * \brief main() for the continuity 2D example
 *
 * Demonstration of simple PDE definition.
 * The file can be used either direcly from the command line
 * or though a proivded simple \ref cont2d_python_code "python driver",
 * which runs the pde and plots the solution.
 *
 * \snippet continuity_2d.cpp continuity_2d main
 *
 * \anchor cont2d_python_code
 * Example Python code that runs the continuity_2d example and plots
 * the solution using the ASGarD Python module.
 *
 * \snippet continuity_2d.py continuity_2d python
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_2d main]
#endif

  // process the command-line arguments and all ASGarD options
  asgard::prog_opts options(argc, argv);

  // raise an error if unknown command line arguments are present
  options.throw_if_invalid();

  // if the user asks for help, print a description of this file
  // and the accepted command line options
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f = s(t, x)\n";
    std::cout << " with periodic boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    return 0;
  }

  // make the 2d domain
  asgard::pde_domain<precision> domain({{-1, 1}, {-2, 2}});

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {5, };

  // compute the max-number of cells in the domain
  int const max_level = options.max_level();

  // smallest cell size that we can have
  precision const dx = domain.min_cell_size(max_level);

  // the cfl condition is that dt < stability-region * dx
  // RK3 stability region is 0.1
  options.default_dt = 0.5 * 0.1 * dx;

  // the time funcitons sin(2 * t) reaches peak at PI / 4
  options.default_stop_time = PI / 4;

  // title and subtitle are usefult to keep track of multiple files and problems
  options.set_default_title("Example continuity 2D");

  // creates a pde description
  asgard::PDEv2<precision> pde(options, domain);

  // one dimensional divergence term using upwind flux
  asgard::term_1d<precision> div =
      asgard::term_div(precision{1},
                       asgard::flux_type::upwind,
                       asgard::boundary_type::periodic);

  asgard::term_1d<precision> I = asgard::term_identity{};

  // 2D divergence consists of two separable multidimensional terms
  pde += asgard::term_md({div, I});
  pde += asgard::term_md({I, div});

  // note: not providing initial conditions implicitly sets them to zero

  // exact solution in x, y and t
  // the functions in x and y are evaluated in batches, hence the vector signature
  // the time function is evaluated one entry at a time
  auto exact_x = [](std::vector<precision> const &x, precision /* time */,
                    std::vector<precision> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::cos(PI * x[i]);
    };
  auto exact_y = [](std::vector<precision> const &y, precision /* time */,
                    std::vector<precision> &fy) ->
    void {
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = std::sin(2 * PI * y[i]);
    };

  auto exact_t = [](precision t) -> precision { return std::sin(2 * t); };

  // the right-hand-sources will also need the derivatives in x, y and t
  auto exact_dx = [](std::vector<precision> const &x, precision /* time */,
                     std::vector<precision> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = - PI * std::sin(PI * x[i]);
    };
  auto exact_dy = [](std::vector<precision> const &y, precision /* time */,
                     std::vector<precision> &fy) ->
    void {
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = 2 * PI * std::cos(2 * PI * y[i]);
    };

  auto exact_dt = [](precision t) -> precision { return 2 * std::cos(2 * t); };

  // the sources take the 3 derivatives
  pde.add_source(asgard::separable_func<precision>{{exact_dx, exact_y},  exact_t});
  pde.add_source(asgard::separable_func<precision>{{exact_x,  exact_dy}, exact_t});
  pde.add_source(asgard::separable_func<precision>{{exact_x,  exact_y},  exact_dt});

  // after the definition of the pde is complete
  // we copy or move the object into the discretization_manager
  asgard::discretization_manager<precision> disc(pde, asgard::verbosity_level::high);

  // solves the pde
  asgard::simulate(disc);

#ifndef __ASGARD_DOXYGEN_SKIP
//! [continuity_2d main]
#endif
  return 0;
}
