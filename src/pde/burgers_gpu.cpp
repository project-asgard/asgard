#include "asgard.h" // alias for asgard.hpp

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file burgers_gpu.cpp
 * \brief Burgers' equation on the GPU
 * \author The ASGarD Team
 *
 * Simple example of non-linear Burgers equation using CUDA or ROCM.
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_burgersgpu Example: GPU version of the Burgers' problem
 *
 * \par Burgers' equation
 * See \ref asgard_examples_burgers "Example: Burgers' non-linear equation",
 * this is using the 2D setup but also enables the use of interpolation functions
 * that call GPU-kernels written in Nvidia CUDA or AMD HIP programming languages.
 *
 * \par
 * This examples shows how to use GPU kernels when setting up interpolation operators.
 */

/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief Indicates the field operation being performed
 *
 * The Burgers' equation uses f^2 as the non-linear operator but in various parts
 * of the code, that has to be split into positive and negative components.
 * See the \ref asgard_examples_burgers "CPU example" for details.
 */
enum class coefficient_mode {
  //! picking up the positive part of the coefficient
  positive,
  //! picking up the negative part of the coefficient
  negative,
  //! take the entire coefficient, i.e., no positive/negative split
  full,
  //! set the source for the inviscit case
  source_inv,
  //! set the source for the diffusive case
  source_diff
};

#if defined(ASGARD_USE_CUDA) || defined(ASGARD_USE_ROCM)
/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief Compute the positive component of the non-linear operator
 *
 * The GPU calculations use raw-array into the function signature. This allows for most
 * generality and does not force a custom (non-std-standard) vector type into the user code.
 *
 * This is a reference CPU implementation of the GPU-kernel above
 *
 * \tparam P is the precision double or float
 *
 * \param time is the current time, currently not implemented
 *
 * \param num_points is the length of f and vals as well as the number of points
 *                   in the nodes[] array
 * \param nodes is an array of size num_points X num_dimensions, in this example, the number
 *        of dimensions are implicitly set to 2, so the (x, y) coordinates for the i-th node
 *        are located at (nodes[2*i], nodes[2*i + 1])
 *
 * \param f has size num_points and the i-th entry will contain the values of the field
 *        at the i-th node
 *
 * \param vals has size num_points and must be overwritten so that the the i-th entry
 *        corresponds to the value of the function at the i-th node
 *
 * \snippet burgers_gpu.cpp burgers f2_kernel
 */
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu f2_kernel]
#endif
template<coefficient_mode mode, typename P>
__global__ void fsquared_kernel(int64_t const num_points, P time,
                                P const nodes[], P const f[], P vals[],
                                P const nu = 0)
{
  (void) time;  // this line just suppresses compiler warnings
  (void) nodes; // this line just suppresses compiler warnings

  // index of the first point processed by this thread
  // this assumes a 1D logical grid of thread-blocks
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < num_points)
  {
    if constexpr (mode == coefficient_mode::positive)
      vals[i] = (f[i] > 0) ? f[i] * f[i] : 0;
    else if constexpr (mode == coefficient_mode::negative)
      vals[i] = (f[i] < 0) ? f[i] * f[i] : 0;
    else if constexpr (mode == coefficient_mode::full)
      vals[i] = f[i] * f[i];
    else if constexpr (mode == coefficient_mode::source_inv)
    {
      auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
      auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };

      auto icy   = [](P y) -> P { return (P{1} - y * y); };
      auto icdy  = [](P y) -> P { return -2 * y; };

      P const x = nodes[2 * i];
      P const y = nodes[2 * i + 1];
      // linear contribution
      vals[i] = -exp(-time) * icx(x) * icy(y);
      // non-linear contribution
      vals[i] += exp(-time) * exp(-time)
                    * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
    }
    else if constexpr (mode == coefficient_mode::source_diff)
    {
      auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
      auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };
      auto icdxx = [](P) -> P { return - P{0.5}; };

      auto icy   = [](P y) -> P { return (P{1} - y * y); };
      auto icdy  = [](P y) -> P { return -2 * y; };
      auto icdyy = [](P) -> P { return -2; };

      P const x = nodes[2 * i];
      P const y = nodes[2 * i + 1];
      // linear contribution
      vals[i] = exp(-time)
               * (-icx(x) * icy(y) - nu * icdxx(x) * icy(y) - nu * icx(x) * icdyy(y));
      // non-linear contribution
      vals[i] += exp(-time) * exp(-time)
                * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
    }

    // move to the next point
    i += blockDim.x * gridDim.x;
  }
}
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu f2_kernel]
#endif
#endif

/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief Compute the positive component of the non-linear operator
 *
 * The GPU calculations use raw-array into the function signature. This allows for most
 * generality and does not force a custom (non-std-standard) vector type into the user code.
 *
 * This is a reference CPU implementation of the GPU-kernel above
 *
 * \tparam P is the precision double or float
 *
 * \param num_points is the length of f and vals as well as the number of points
 *                   in the nodes[] array
 * \param nodes is an array of size num_points X num_dimensions, in this example, the number
 *        of dimensions are implicitly set to 2
 *
 * \param f is the field, used by the coeffieints and ignored by the sources
 *
 * \param vals is the output array of size num_points that will hold the result
 *
 * \param nu is the collision frequency, used in the non-inviscit source
 *
 * \snippet burgers_gpu.cpp burgers f2pos
 */
template<coefficient_mode mode, typename P>
void fsquared(int64_t const num_points, P time, P const nodes[], P const f[], P vals[], P nu = 0)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu f2pos]
#endif
  // this is a demonstration of now a kernel launch can be incorporated with ASGarD
  // this is not a tutorial on how to write CUDA/ROCM/HIP kernels

  // the kernel will launch on a one-dimensional thread grid and 1 thread per point

  // setting up the number of threads in a thead-block
  int constexpr num_threads = 1024;

  // how many thread blocks do we need for the given num_points
  // the operation rounds up, so that num_blocks * num_threads >= num_points
  int const num_blocks = (num_points + num_threads - 1) / num_threads;

  // call a CUDA/ROCM kernel
  #if defined(ASGARD_USE_CUDA) || defined(ASGARD_USE_ROCM)
  fsquared_kernel<mode, P><<<num_blocks, num_threads>>>(num_points, time,nodes, f, vals, nu);
  #endif

#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu f2pos]
#endif
}

/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief Make an Burger's PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param num_dims number of dimensions
 * \param options is the set of options
 *
 * \returns a pde_scheme<P> set for the Burgers equation
 *
 * \snippet burgers_gpu.cpp burgers make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_burgers_pde(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu make]
#endif

  options.title = "Burgers PDE 2D";

  std::optional<P> const cli_nu = options.extra_cli_value_group<P>({"-nu", });
  if (not cli_nu and options.is_mpi_rank_zero())
    std::cout << "no '-nu' provided, defaulting to inviscit Burgers '-nu 0'\n";

  P const nu = cli_nu.value_or(0);

  if (nu < 0)
    throw std::runtime_error("the viscosity coefficient '-nu' should be non-negative");

  if (nu == 0)
    options.title += " (inviscit)";
  else
    options.title += " (viscosity nu = " + std::to_string(nu) + ")";

  asgard::pde_domain<P> domain
    = asgard::pde_domain<P>(std::vector<asgard::domain_range>(2, {-1.0, 1.0}));

  options.default_degree = 3;
  options.default_start_levels = {6, };

  // the inviscit equation can be done with an explicit time stepper
  if (nu == 0)
    options.default_step_method = asgard::time_method::rk2;
  else
    options.default_step_method = asgard::time_method::imex1;

  P const dx = domain.min_cell_size(options.max_level());
  options.default_dt = 0.05 * dx;
  options.default_stop_time = 0.5;

  options.default_solver = asgard::solver_method::bicgstab;

  // defaults for iterative solvers, not necessarily optimal
  options.default_isolver_tolerance  = 1.E-8;
  options.default_isolver_iterations = 1000;

  options.default_precon = asgard::precon_method::jacobi;

  asgard::pde_scheme<P> pde(options, std::move(domain));

  #if defined(ASGARD_USE_CUDA) || defined(ASGARD_USE_ROCM)
  auto f2p = [=](int64_t num_points, P t, P const x[], P const f[], P vals[]) ->
    void {
      fsquared<coefficient_mode::positive, P>(num_points, t, x, f, vals);
    };
  auto f2n = [=](int64_t num_points, P t, P const x[], P const f[], P vals[]) ->
    void {
      fsquared<coefficient_mode::negative, P>(num_points, t, x, f, vals);
    };

  // ensure that the adaptive process captures the nonlinear component in addition to the field
  auto f2 = [=](int64_t num_points, P t, P const x[], P const f[], P vals[]) ->
    void {
      fsquared<coefficient_mode::full, P>(num_points, t, x, f, vals);
    };
  #else
  // if CUDA/ROCM are not enabled, defaulting to the CPU
  auto f2p = [=](P, asgard::vector2d<P> const &,
                 std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = (f[i] > 0) ? f[i] * f[i] : 0;
      }
    };
  auto f2n = [=](P, asgard::vector2d<P> const &,
                 std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = (f[i] < 0) ? f[i] * f[i] : 0;
      }
    };

  auto f2 = [=](P, asgard::vector2d<P> const &,
                std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = f[i] * f[i];
      }
    };
  #endif

  auto f2cpu = [=](P, asgard::vector2d<P> const &,
                std::vector<P> const &f, std::vector<P> &vals) ->
    void {
      for (size_t i = 0; i < f.size(); i++) {
        vals[i] = f[i] * f[i];
      }
    };
  pde.set_adapt_weight(f2cpu);

  // setting up multidimensional volume term that uses interpolated coefficient
  // the signature of the f2p and f2n functions determine whether to use the CPU
  // or the GPU device
  asgard::term_md<P> term_f2_pos = asgard::term_interp<P>{f2p};
  asgard::term_md<P> term_f2_neg = asgard::term_interp<P>{f2n};

  // initial conditions and derivatives in x and y, also the exact solution in time
  auto icx   = [](P x) -> P { return P{1} + P{0.75} * x - P{0.25} * x * x; };
  auto icdx  = [](P x) -> P { return P{0.75} - P{0.5} * x; };
  auto icdxx = [](P) -> P { return - P{0.5}; };

  auto icy   = [](P y) -> P { return (P{1} - y * y); };
  auto icdy  = [](P y) -> P { return -2 * y; };
  auto icdyy = [](P) -> P { return -2; };

  auto exact_t = [](P t) -> P { return std::exp(-t); };

  if (nu == 0) {
    // inviscit mode, using explicit time-stepping and no second order terms
    // using the default flux_type::upwind and boundary_type::none
    asgard::term_md<P> divx_pos = {asgard::term_div<P>{0.5, asgard::boundary_type::left},
                                   asgard::term_identity{}};
    asgard::term_md<P> divx_neg = {asgard::term_div<P>{0.5, asgard::flux_type::downwind},
                                   asgard::term_identity{}};
    asgard::term_md<P> divy_pos = {asgard::term_identity{},
                                   asgard::term_div<P>{0.5, asgard::boundary_type::left}};
    asgard::term_md<P> divy_neg = {asgard::term_identity{},
                                   asgard::term_div<P>{0.5, asgard::boundary_type::right,
                                                       asgard::flux_type::downwind}};

    pde += asgard::term_md<P>{divx_pos, term_f2_pos};
    pde += asgard::term_md<P>{divx_neg, term_f2_neg};
    pde += asgard::term_md<P>{divy_pos, term_f2_pos};
    pde += asgard::term_md<P>{divy_neg, term_f2_neg};

    // setting up the non-separable source
    #if defined(ASGARD_USE_CUDA) || defined(ASGARD_USE_ROCM)
    auto smd = [=](int64_t num_points, P t, P const x[], P vals[]) ->
      void {
        fsquared<coefficient_mode::source_inv, P>(num_points, t, x, nullptr, vals);
      };
    #else
    auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
      void {
        for (int64_t i = 0; i < nodes.num_strips(); i++) {
          P const x = nodes[i][0];
          P const y = nodes[i][1];
          // linear contribution
          vals[i] = -std::exp(-t) * icx(x) * icy(y);
          // non-linear contribution
          vals[i] += std::exp(-t) * std::exp(-t)
                    * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
        }
      };
    #endif

    // a term-group can have at most one non-separable source
    // thus we use the "set" method, as opposed to "add"
    pde.set_source(smd);

  } else {
    // boundary conditions in y are homogeneous and simple to impose to all terms
    // boundary conditions in x are imposed only on the second order term
    asgard::term_md<P> divx_pos = {asgard::term_div<P>{0.5, asgard::flux_type::upwind},
                                   asgard::term_identity{}};
    asgard::term_md<P> divx_neg = {asgard::term_div<P>{0.5, asgard::flux_type::downwind},
                                   asgard::term_identity{}};
    asgard::term_md<P> divy_pos = {asgard::term_identity{},
                                   asgard::term_div<P>{0.5, asgard::boundary_type::left,
                                                       asgard::flux_type::upwind}, };
    asgard::term_md<P> divy_neg = {asgard::term_identity{},
                                   asgard::term_div<P>{0.5, asgard::boundary_type::right,
                                                       asgard::flux_type::downwind}, };

    // the group ids are needed for IMEX scheme in the viscous way
    int const non_linear_group_id = pde.new_term_group();

    pde += asgard::term_md<P>{divx_pos, term_f2_pos};
    pde += asgard::term_md<P>{divx_neg, term_f2_neg};
    pde += asgard::term_md<P>{divy_pos, term_f2_pos};
    pde += asgard::term_md<P>{divy_neg, term_f2_neg};

    // setting up the non-separable source
    // #if defined(ASGARD_USE_CUDA) || defined(ASGARD_USE_ROCM)
    // auto smd = [=](int64_t num_points, P t, P const x[], P vals[]) ->
    //   void {
    //     fsquared<coefficient_mode::source_diff, P>(num_points, t, x, nullptr, vals, nu);
    //   };
    // #else
    auto smd = [=](P t, asgard::vector2d<P> const &nodes, std::vector<P> &vals) ->
      void {
        for (int64_t i = 0; i < nodes.num_strips(); i++) {
          P const x = nodes[i][0];
          P const y = nodes[i][1];
          // linear contribution
          vals[i] = std::exp(-t)
                   * (-icx(x) * icy(y) - nu * icdxx(x) * icy(y) - nu * icx(x) * icdyy(y));
          // non-linear contribution
          vals[i] += std::exp(-t) * std::exp(-t)
                    * (icx(x) * icdx(x) * icy(y) * icy(y) + icx(x) * icx(x) * icy(y) * icdy(y));
        }
      };
    // #endif

    // setting the non-separable source into the pde_scheme
    pde.set_source(smd);

    // second order term in x
    asgard::term_1d<P> div_grad_x = std::vector<asgard::term_1d<P>>{
        asgard::term_div<P>{-std::sqrt(nu), asgard::boundary_type::none},
        asgard::term_grad<P>{std::sqrt(nu), asgard::boundary_type::bothsides},
      };

    div_grad_x.set_penalty(1 / dx);

    asgard::term_md<P> dgx = {div_grad_x, asgard::term_identity{}};

    // adding inhomogeneous boundary condition on the right
    asgard::separable_func<P> fr(std::vector<P>{icx(pde.domain().xright(0)), 1}, exact_t);
    fr.set(asgard::dimension_id{1},
           [=](std::vector<P> const &y, P, std::vector<P> &fy) ->
              void {
                for (size_t i = 0; i < y.size(); i++)
                  fy[i] = icy(y[i]);
              });
    dgx += asgard::right_boundary_flux{fr};

    asgard::term_1d<P> div_grad_y = std::vector<asgard::term_1d<P>>{
        asgard::term_div<P>{-std::sqrt(nu), asgard::boundary_type::none},
        asgard::term_grad<P>{std::sqrt(nu), asgard::boundary_type::bothsides},
      };

    div_grad_y.set_penalty(1 / dx);

    asgard::term_md<P> dgy = {asgard::term_identity{}, div_grad_y};

    // adding the second order terms to a new term-group
    int const laplacian_group_id = pde.new_term_group();
    pde += dgx;
    pde += dgy;

    pde.set(asgard::imex_implicit_group{laplacian_group_id},
            asgard::imex_explicit_group{non_linear_group_id});
  }

  // the vector version of the initial conditions
  auto icx_vec = [=](std::vector<P> const &x, P, std::vector<P> &fx)
    -> void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = icx(x[i]);
    };
  auto icy_vec = [=](std::vector<P> const &y, P, std::vector<P> &fy)
    -> void {
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = icy(y[i]);
    };

  pde.add_initial(asgard::separable_func<P>({icx_vec, icy_vec}, exact_t));

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu make]
#endif
}

/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief Computes the L^2 error for the given example
 *
 * The provided discretization_manager should hold a PDE made with
 * make_burgers_pde(). This will compute the L^2 error.
 *
 * \tparam P is double or float, the precision of the manager
 *
 * \param disc is the discretization of a PDE
 *
 * \returns the L^2 error between the known exact solution and
 *          the current state in the \b disc manager
 *
 * \snippet burgers_gpu.cpp burgers_gpu get-err
 */
template<typename P>
double get_error_l2(asgard::discretization_manager<P> const &disc) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu get-err]
#endif

  // using the fact that the initial condition is the exact solution
  std::vector<P> const eref = disc.project_function(disc.initial_cond_sep());

  double constexpr space = 1984.0 / 900.0;
  double const time_val  = std::exp(-disc.time());

  // this is the L^2 norm-squared of the exact solution
  double const enorm = space * time_val * time_val;

  std::vector<P> const &state = disc.current_state_mpi();
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
//! [burgers_gpu get-err]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
// internal testing, not part of the example
void self_test();
#endif

/*!
 * \ingroup asgard_examples_burgersgpu
 * \brief main() for the continuity example
 *
 * The main() processes the command line arguments and calls both
 * make_burgers_pde() and get_error_l2().
 *
 * \snippet burgers_gpu.cpp burgers_gpu main
 */
int main(int argc, char **argv) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu main]
#endif

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this file and the two additional options accepted for this problem
  if (options.show_help) {
    std::cout << "\n solves the continuity equation:\n";
    std::cout << "    f_t + div f^2 = nu * f_xx + s(t, x)\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-nu                                 diffusion coefficient\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  options.throw_if_argv_not_in({"-test", }, {"-nu", });

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // creating a discretization manager
  asgard::discretization_manager<P> disc(make_burgers_pde<P>(options),
                                         asgard::verbosity_level::high);

  P const err_init = get_error_l2(disc);
  if (not disc.stop_verbosity())
    std::cout << " -- error in the initial conditions: " << err_init << "\n";

  disc.advance_time();

  if (not disc.stop_verbosity())
    disc.progress_report();

  disc.save_final_snapshot();

  P const err_final = get_error_l2(disc);
  if (not disc.stop_verbosity()) {
    disc.progress_report();
    std::cout << " -- final error: " << err_final << "\n";
  }

  if (asgard::tools::timer.enabled() and not disc.stop_verbosity())
    std::cout << asgard::tools::timer.report() << '\n';

  return 0;
#ifndef __ASGARD_DOXYGEN_SKIP
//! [burgers_gpu main]
#endif
}

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////
using namespace asgard;

template<typename P>
void dotest(double tol, std::string const &opts) {
  current_test<P> test_(opts, 2);

  auto options = make_opts(opts);

  discretization_manager<P> disc(make_burgers_pde<P>(options),
                                 verbosity_level::quiet);

  while (disc.remaining_steps() > 0)
  {
    disc.advance_time(1);

    double const err = get_error_l2(disc);

    tcheckless(disc.current_step(), err, tol);
  }
}

void self_test() {
  all_tests testing_("Burgers' equation:", " using CUDA or ROCM");

#ifdef ASGARD_ENABLE_DOUBLE
  dotest<double>(2.E-5, "-l 6 -n 20 -nu 0.1");
  dotest<double>(2.E-5, "-l 6 -n 20 -nu 0.1 -s imex2");
  dotest<double>(1.E-7, "-l 6 -n 20 -nu 0");
  dotest<double>(1.E-7, "-l 3 -m 8 -n 20 -a 1.E-8 -nu 0");
#endif

#ifndef ASGARD_ENABLE_DOUBLE
  dotest<float>(2.E-3, "-l 6 -n 20 -nu 0");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
