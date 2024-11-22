#include "asgard.hpp"

// if ASGarD is compiled with double precision, this defaults to double
// if only single precision is avaiable, this will be float
using precision = asgard::default_precision;

// this problem is identical to PDE_continuity2d
// the only difference is that it is set as a stand-alone example
// and it is not incorporated into the core library
class example_continuity2d : public asgard::PDE<precision>
{
public:
  // short-hand notation of the asgard vector
  using vector = asgard::fk::vector<precision>;
  // short-hand notation for the asgard 1d dimension
  using dimension = asgard::dimension<precision>;

  // short-hand notations
  // a collection of PDE operators (term_set) is broken up into separable
  //   multidimensional terms (term_md)
  // each multidimensional terms has num-dimensions 1d components (term1d)
  // each one dimensional term consists of one or more partial terms
  using partial_term_1d = asgard::partial_term<precision>;
  using term_1d = asgard::term<precision>;
  using term_md = std::vector<asgard::term<precision>>;
  // term_set is actually std::vector<term_md>
  using term_set = asgard::term_set<precision>;

  // short-hand notation
  using source_md  = asgard::source<precision>;
  using source_set = std::vector<source_md>;

  example_continuity2d(asgard::prog_opts const &cli_input)
  {
    // these fields check correctness of the specification
    int constexpr num_dimensions = 2;
    int constexpr num_sources    = 3;
    int constexpr num_terms      = 2;

    // the Poisson solver is not used here
    bool constexpr do_poisson_solve  = false;

    // flagging terms as time_independent (using the false value)
    // improves speed by keeping some constant matrices across time-steps
    bool constexpr time_independent = false;

    // define the two dimensional domain and provide the initial conditions
    // the level and degree can be modified via the command line arguments
    dimension dim0(-1.0,                // domain min
                   1.0,                 // domain max
                   2,                   // levels
                   2,                   // degree
                   initial_condition_x, // initial condition
                   nullptr,             // Cartesian coordinates
                   "x");                // reference name

    dimension dim1(-2.0,                // domain min
                   2.0,                 // domain max
                   2,                   // levels
                   2,                   // degree
                   initial_condition_y, // initial condition
                   nullptr,             // Cartesian coordinates
                   "y");                // reference name

    // building up the divergence operator from simple components
    partial_term_1d par_derivative(
        asgard::coefficient_type::div, // uses derivative
        op_coeff,                      // -1.0 (appears on the r.h.s)
        nullptr,                       // no l.h.s. coefficient
        asgard::flux_type::upwind,
        asgard::boundary_condition::periodic,
        asgard::boundary_condition::periodic);

    partial_term_1d par_mass(
        asgard::coefficient_type::mass, // mass matrix, identity
        nullptr,                        // no r.h.s. coefficient
        nullptr,                        // no l.h.s. coefficient
        asgard::flux_type::central,
        asgard::boundary_condition::periodic,
        asgard::boundary_condition::periodic);

    term_1d d_x(time_independent, "d_x", {par_derivative});
    term_1d mass_y(time_independent, "mass_y", {par_mass});

    term_1d mass_x(time_independent, "mass_x", {par_mass});
    term_1d d_y(time_independent, "d_y", {par_derivative});

    term_set terms = {
        term_md{d_x, mass_y},
        term_md{mass_x, d_y}
    };

    // the PDE has analytic solution
    // the computed solution will be verified against the analytic one
    // the solution is the product of the the 3 functions
    static bool constexpr has_analytic_solution = true;
    std::vector<asgard::vector_func<precision>> exact_solution = {
        initial_condition_x, initial_condition_y, exact_time_vector};

    // in order to manufacture a specific analytic solution
    // we use artificial source terms that balance the derivatives
    // see the comments before the definition of diff_time
    // each separable source consists of num-dimensions spacial funcitons
    // and a time function
    source_md s0({initial_condition_dx, initial_condition_y},  exact_solution_time);
    source_md s1({initial_condition_x,  initial_condition_dy}, exact_solution_time);
    source_md s2({initial_condition_x,  initial_condition_y},  exact_solution_dt);

    source_set sources = {s0, s1, s2};

    // once all the components are prepared, the PDE must be initialized
    // this is done at the end of the constructor
    this->initialize(
        cli_input, // allows modifications, e.g., override mesh level
        num_dimensions, num_sources, num_terms,  // for sanity-check purposes
        std::vector<dimension>{dim0, dim1}, // domain
        terms, sources, exact_solution,
        get_dt, do_poisson_solve, has_analytic_solution);
  }

private:
  //
  // function definitions needed to build up the dimension, terms and sources
  //

  // for all functions, the "vector x" indicates a batch of quadrature points
  // in the corresponding dimension (e.g., dim0 or dim1)
  // the output should be a vector with the same size holding f(x)
  // functions also accept a "time" scalar but it is often ignored

  // specify initial condition vector functions...
  static vector initial_condition_x(vector const &x, precision const = 0)
  {
    // ignored parameter corresponds to time
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::cos(M_PI * x[i]);
    return fx;
  }

  static vector initial_condition_y(vector const &x, precision const = 0)
  {
    // ignored parameter corresponds to time
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::sin(precision{2.0} * M_PI * x[i]);
    return fx;
  }

  // specify exact solution, which is
  // initial_condition_dim0 * initial_condition_dim1 * exact_solution_time

  // The exact_solution_time() has two signatures:
  // - scalar-to-scalar signature used in the source definitions
  // - vector signature that matches the signature of the initial conditions
  static precision exact_solution_time(precision const t)
  {
    return std::sin(precision{2.0} * t);
  }

  static vector exact_time_vector(vector const &, precision const t = 0)
  {
    // unlike the initial condition functions, the time variable is used
    // while the x-variable is ignored
    return {
        exact_solution_time(t),
    };
  }

  // specify source functions...
  // we are using the method of manufactured solutions where we create
  // a source that will balance the PDE terms and result in problem
  // with known analytic solution

  // to this end, we need the derivatives
  // of the three components of the exact solution

  static vector initial_condition_dx(vector const &x, precision const = 0)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = - M_PI * std::sin(M_PI * x[i]);
    return fx;
  }

  static vector initial_condition_dy(vector const &x, precision const = 0)
  {
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = precision{2.0} * M_PI * std::cos(precision{2.0} * M_PI * x[i]);
    return fx;
  }

  // derivative of exact_solution_time
  static precision exact_solution_dt(precision const t)
  {
    return precision{2.0} * std::cos(precision{2.0} * t);
  }

  // a bit of a misnomer, this is a scaling factor applied to dt
  // in conjunction with the CFL condition provided from the command line
  // (note: this is probably not fully supported at the moment)
  static precision get_dt(asgard::dimension<precision> const &dim)
  {
    precision const x_range = dim.domain_max - dim.domain_min;
    precision const dx      = x_range / asgard::fm::ipow2(dim.get_level());
    // return dx; this will be scaled by CFL
    // from command line
    return dx;
  }

  // operator coefficient
  static precision op_coeff(precision const, precision const)
  {
    // the signature is op_coeff(x, t) where x is in space and t is time
    // here we are using a simple operator with just -1.0
    return -1.0;
  }

};

int main(int argc, char **argv)
{
  // if using MPI, this will call MPI_Init()
  // with or without MPI, this sets the ASGarD environment
  auto const [my_rank, num_ranks] = asgard::initialize_distribution();

  // kill off unused processes (MPI only)
  if (my_rank >= num_ranks)
  {
    asgard::finalize_distribution();
    return 0;
  }

  // parse the command line inputs
  asgard::prog_opts const options(argc, argv);

  // main call to asgard, does all the work
  // i.e., create an instance of the PDE that we want to solve, discretizes it
  // and advances the time according to the options and the example_continuity2d
  // specification
  // also, outputs snapshots of the solution
  asgard::simulate<example_continuity2d>(options, asgard::verbosity_level::high);

  // call MPI_Finalize() and/or cleanup after the simulation
  asgard::finalize_distribution();

  return 0;
}
