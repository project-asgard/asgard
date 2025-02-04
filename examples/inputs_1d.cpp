#include "asgard.hpp"

/*!
 * \internal
 * \file inputs_1d.cpp
 * \brief Simple 1D example
 * \author The ASGarD Team
 * \ingroup asgard_examples_input_1d
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_input_1d Example 2, Simple 1D equation
 *
 * \par Example 2
 * Solves the continuity partial differential equation
 * \f[ \frac{\partial}{\partial t} f + \frac{\partial}{\partial x} f = s \f]
 * where both \b f and \b s are defined over domain
 * \f[ (-\pi N_w, \pi N_w) \f]
 * where N-w is the number of waves.
 *
 * \par
 * The right right-hand-side source is chosen so the exact solution
 * is a simple separable function
 * \f[ f(t, x) = \cos(t) \sin(x) \f]
 *
 * \par
 * This example demonstrates the use of a input file to set problem parameters.
 * Two example input files are included and a companion Python script.
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
 * \ingroup asgard_examples_input_1d
 * \brief The ratio of circumference to diameter of a circle
 */
precision constexpr PI = asgard::PI;

/*!
 * \ingroup asgard_examples_input_1d
 * \brief main() for the input 1D example
 *
 * Demonstration of simple PDE definition, where some of the problem
 * parameters are read from an input file.
 * The example comes with a \ref inp1d_python_code "companion Python script".
 *
 * \snippet inputs_1d.cpp inputs_1d main
 *
 * \anchor inp1d_python_code
 * Example Python code that runs the inputs_1d example and plots
 * the solution using the ASGarD Python module.
 *
 * \snippet inputs_1d.py inputs_1d python
 */
int main(int argc, char **argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [inputs_1d main]
#endif

  // process the command-line arguments and all ASGarD options
  asgard::prog_opts options(argc, argv);

  // raise an error if unknown command line arguments are present
  options.throw_if_invalid();

  // we expect to find a file with "number of waves" defined in it
  std::optional<int> opt_num_waves = options.file_value<int>("number of waves");
  if (not opt_num_waves)
    throw std::runtime_error("inputs_1d needs an input file with "
                             "the 'number of waves' defined in it");

  // alternative to the check above
  // int const num_waves = options.file_required<int>("number of waves");
  // the "file_required" method will throw if the value is missing
  int const num_waves = opt_num_waves.value();

  // if the user asks for help, print a description of this file
  // and the accepted command line options
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + f_x = s(t, x)\n";
    std::cout << " with periodic boundary conditions \n"
                 " and source term that generates a known artificial solution\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    return 0;
  }

  // make the 1d domain
  asgard::pde_domain domain({{-PI * num_waves, PI * num_waves}, });

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  options.default_degree = 2;
  options.default_start_levels = {5, };

  // compute the max-number of cells in the domain
  int const max_level = options.max_level();

  // smallest cell size that we can have
  double const dx = domain.min_cell_size(max_level);

  // the cfl condition is that dt < stability-region * dx
  // RK3 stability region is 0.1
  options.default_dt = 0.5 * 0.1 * dx;

  // the time funcitons cos(t) reaches negative max at PI
  options.default_stop_time = PI;

  // title and subtitle are usefult to keep track of multiple files and problems
  options.set_default_title("Example inputs 1D");

  // creates a pde description
  asgard::PDEv2 pde(options, domain);

  // one dimensional divergence term using upwind flux
  pde += asgard::term_1d{asgard::term_div(1, asgard::flux_type::upwind,
                                             asgard::boundary_type::periodic)};

  // exact solution
  auto exact_x = [](std::vector<precision> const &x, precision /* time */,
                    std::vector<precision> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::sin(x[i]);
    };

  auto exact_t = [](precision t) -> precision { return std::cos(t); };

  // derivatives of the components
  auto exact_dx = [](std::vector<precision> const &x, precision /* time */,
                     std::vector<precision> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = std::cos(x[i]);
    };

  auto exact_dt = [](precision t) -> precision { return -std::sin(t); };

  // using the exact solution as initial condition
  pde.add_initial(asgard::separable_func<precision>{{exact_x, }, exact_t});

  // the sources take the 3 derivatives
  pde.add_source(asgard::separable_func<precision>{{exact_x,  }, exact_dt});
  pde.add_source(asgard::separable_func<precision>{{exact_dx, }, exact_t});

  // after the definition of the pde is complete
  // we copy or move the object into the discretization_manager
  asgard::discretization_manager<precision> disc(pde, asgard::verbosity_level::high);

  // solves the pde
  asgard::simulate(disc);

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [inputs_1d main]
#endif
}
