#include "asgard.hpp"

// if ASGarD is compiled with double precision, this defaults to double
// if only single precision is avaiable, this will be float
using precision = asgard::default_precision;

// very simple PDE that demonstrates using external file with parameters
//   df / dt + df / dx = s(t, x)
// the domain is set to be (-n * M_PI, n * M_PI)
// here n is set to be the number of waves, i.e., num_waves static variable
// the forcing term is chosen so the exact solution is just sin(x) cos(t)
class example_sinwaves : public asgard::PDE<precision>
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

  example_sinwaves(asgard::prog_opts const &cli_input)
  {
    // these fields check correctness of the specification
    int constexpr num_dimensions = 1;
    int constexpr num_sources    = 2;
    int constexpr num_terms      = 1;

    // the Poisson solver is not used here
    bool constexpr do_poisson_solve  = false;

    // flagging terms as time_independent (using the false value)
    // improves speed by keeping some constant matrices across time-steps
    bool constexpr time_independent = false;

    // define the domain but use the static variable num_waves
    if (num_waves <= 0)
      throw std::runtime_error("num_waves must be set to a positive integer");

    dimension dim0(-M_PI * num_waves,   // domain min
                   M_PI * num_waves,    // domain max
                   2,                   // levels
                   2,                   // degree
                   initial_condition_x, // initial condition
                   nullptr,             // Cartesian coordinates
                   "x");                // reference name

    // building up the df / dx term
    partial_term_1d par_derivative(
        asgard::coefficient_type::div, // uses derivative
        op_coeff,                      // -1.0 (appears on the r.h.s)
        nullptr,                       // no l.h.s. coefficient
        asgard::flux_type::upwind,
        asgard::boundary_condition::periodic,
        asgard::boundary_condition::periodic);

    term_1d d_x(time_independent, "d_x", {par_derivative});

    term_set terms = { term_md{d_x, }, };

    // the PDE has analytic solution
    // the computed solution will be verified against the analytic one
    // the solution is the product of the the 3 functions
    static bool constexpr has_analytic_solution = true;
    std::vector<asgard::vector_func<precision>> exact_solution = {
        initial_condition_x, exact_time_vector};

    // in order to manufacture a specific analytic solution
    // we use artificial source terms that balance the derivatives
    // see the comments before the definition of diff_time
    // each separable source consists of num-dimensions spacial funcitons
    // and a time function
    source_md s0({initial_condition_dx, },  exact_solution_time);
    source_md s1({initial_condition_x,  },  exact_solution_dt);

    source_set sources = {s0, s1};

    // once all the components are prepared, the PDE must be initialized
    // this is done at the end of the constructor
    this->initialize(
        cli_input, // allows modifications, e.g., override mesh level
        num_dimensions, num_sources, num_terms,  // for sanity-check purposes
        std::vector<dimension>{dim0, }, // domain
        terms, sources, exact_solution,
        get_dt, do_poisson_solve, has_analytic_solution);
  }

  // using a static variable allows us to include the variable into function
  // calls without the need to use closures capturing the "this" pointer
  // in this example, we are only using num_waves in the domain min/max
  // but in general, this class is what is known as a "singleton"
  inline static int num_waves = 0;

private:
  //
  // function definitions needed to build up the dimension, terms and sources
  //

  // for all funtions, the "vector x" indicates a batch of quadrature points
  // in the corresponding dimension (e.g., dim0 or dim1)
  // the output should be a vector with the same size holding f(x)
  // funcitons also accept a "time" scalar but it is often ignored

  // specify initial condition vector functions...
  static vector initial_condition_x(vector const &x, precision const = 0)
  {
    // ignored parameter corresponds to time
    vector fx(x.size());
    for (int i = 0; i < x.size(); i++)
      fx[i] = std::sin(x[i]);
    return fx;
  }

  // The exact_solution_time() has two signatures:
  // - scalar-to-scalar signature used in the source definitions
  // - vector signature that matches the signature of the initial conditions
  static precision exact_solution_time(precision const t)
  {
    return std::cos(t);
  }

  static vector exact_time_vector(vector const &, precision const t)
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
      fx[i] = std::cos(x[i]);
    return fx;
  }

  // derivative of exact_solution_time
  static precision exact_solution_dt(precision const t)
  {
    return -std::sin(t);
  }

  // a bit of a misnomer, this is a scaling factor applied to dt
  // in conjuction with the CFL conidition provided from the command line
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
  asgard::prog_opts options(argc, argv);

  std::optional<int> num_waves = options.file_value<int>("number of waves");
  if (not num_waves)
    throw std::runtime_error("inputs_1d needs an input file with "
                             "the 'number of waves' defined in it");

  // the static variable must be set before we create an instance
  // of the example_sinwaves PDE
  example_sinwaves::num_waves = num_waves.value();

  int cells_per_wavel = 16; // assume we want an accurate wave picture
  std::optional<int> read_cpw = options.file_value<int>("cells per wave");

  if (read_cpw) // if the file contains custom "cells per wave" entry
    cells_per_wavel = read_cpw.value();
  // else, we will use the default 16 set from above

  // setup the discretization parameters, we at least 8 grid cells per wave
  int level     =  4;
  int num_cells = 16; // level 4 yields 16 cells
  while (num_cells < cells_per_wavel * example_sinwaves::num_waves)
  {
    // not enough cells, increase the level
    ++level;
    // number of cells per level is 2^level
    num_cells *= 2;
  }

  // set the fixed grid
  // the code can be adjusted here so we take the max of level
  // and any existing level provided by the input file or cli
  options.start_levels = {level, };

  // we also need to adjust the time-step and number of steps

  // stability region for RK3 time-stepping method <= 0.01, using 0.005
  // CFL condition is equal to the cell size
  // cell size is the domain size 2 * num_waves * M_PI / num_cells

  double cell_size
      = 2.0 * example_sinwaves::num_waves * M_PI
        / static_cast<double>(num_cells);

  options.dt = 0.005 / cell_size;

  // desired final time
  // the code can be adjusted so that time is also read from the file
  // (also the input files have to be adjusted)
  double const time = 1.0;

  // round up to the time-steps that it will take to reach time = 1.0
  options.num_time_steps = 1 + static_cast<int>(time / options.dt.value());

  // note here that dt * num_time_steps is not necessarily 1.0 due to rounding
  // readjust the dt to so the rounding is less
  options.dt = time / static_cast<double>(options.num_time_steps.value());

  // we want to generate a single output file at the end
  options.wavelet_output_freq = options.num_time_steps;

  // create an instance of the PDE that we want to solve and discretize it
  auto disc = asgard::discretize<example_sinwaves>(options,
                                                   asgard::verbosity_level::high);

  // perform the time stepping
  asgard::advance_time(disc);

  // finalize the solution, also outputs the files
  disc.save_final_snapshot();

  // the continuity_2d example uses a "one-shot" method for simulating a pde
  // here, we create an asgard::discretization_manager (disc) which allows for
  // more fine grained control of the time-advance approach
  // e.g., we could break up the time-advance method and save intermediate
  // snapshot of the solution

  if (asgard::get_local_rank() == 0)
    std::cout << " -- done simulating " << example_sinwaves::num_waves << '\n';

  // call MPI_Finalize() and/or cleanup after the simulation
  asgard::finalize_distribution();

  return 0;
}
