#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file landau_damping.cpp
 * \brief Landau damping example
 * \author The ASGarD Team
 * \ingroup asgard_examples_landau
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_landau_damping Example: Landau damping
 *
 * \par Landau damping
 * Solves the Vlasov-Poisson equation in a common example
 * often called the landau damping problem
 * \f[ \frac{\partial}{\partial t} f(x, v) + v \cdot \nabla_x f(x, v, t) + E(x, t) \nabla_v \cdot f(x, v, t) = 0 \f]
 * where the electric field term depends on the Poisson equation
 * \f[ E(x,t) = -\nabla_x \Phi(x, t), \qquad - \nabla_x \cdot \nabla_x \Phi(x, t) = \int_v f(x, v, t) dv \f]
 * The equation represents the evolution of a charged particle field under the effects
 * of self-induced electric field.
 * The right-hand integral represents the density of the particles and creates
 * non-linear coupling between the fields.
 *
 * \par
 * The focus of this example is the coupling with the electric field and Poisson
 * solver.
 *
 * \par
 * <i>This is still work-in-progress, the documentation needs more work.</i>
 */

/*!
 * \ingroup asgard_examples_landau_damping
 * \brief The ratio of circumference to diameter of a circle
 */
double constexpr PI = asgard::PI;

#ifndef __ASGARD_DOXYGEN_SKIP
// self-consistency testing, not part of the example/tutorial
void self_test();
#endif

#ifdef ASGARD_USE_GPU
template<typename P>
__global__ void interp_positive_kernel(int64_t num, P const* field, P const* mom, P* out) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num) {
        // Avoiding std::max to prevent __device__ compilation header conflicts
        out[i] = field[i] * ((mom[i] > 0.0) ? mom[i] : 0.0); 
    }
}

template<typename P>
__global__ void interp_negative_kernel(int64_t num, P const* field, P const* mom, P* out) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num) {
        out[i] = field[i] * ((mom[i] < 0.0) ? mom[i] : 0.0);
    }
}
#endif

/*!
 * \ingroup asgard_examples_landau_damping
 * \brief Make single landau damping PDE
 *
 * Constructs the pde description for the given number of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme definition
 *
 * \snippet landau_damping.cpp landau make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_landau(asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [landau make]
#endif

  options.title = "Multi-D Landau Damping";

  // the domain has one position and one velocity dimension: 1x1v
  asgard::pde_domain<P> domain(asgard::position_dims{2}, asgard::velocity_dims{2},
                               {{-2 * PI, 2 * PI}, {-2 * PI, 2 * PI},
                                {-2 * PI, 2 * PI}, {-2 * PI, 2 * PI}});

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  int const default_degree = 2;

  options.default_degree = default_degree;
  options.default_start_levels = {7, 7, 7, 7};

  options.default_plotter_colormap = "viridis";

  // the CFL is more complicated, it depends both on the polynomial degree
  // and on the maximum number of cells (TODO: add more here)
  int const k = options.degree.value_or(default_degree);
  int const n = (1 << options.max_level());
  options.default_dt = 3.0 / (2 * (2 * k + 1) * n);

  options.default_stop_time = 0.25;

  // using explicit RK2
  options.default_step_method = asgard::time_method::rk2;

  // create a pde from the given options and domain
  asgard::pde_scheme<P> pde(options, domain);

  // terms are split into positive and negative
  auto positive = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::max(P{0}, x[i]);
    };

  // terms are split into positive and negative
  auto negative = [](std::vector<P> const &x, std::vector<P> &y)
      -> void
    {
#pragma omp parallel for
      for (size_t i = 0; i < x.size(); i++)
        y[i] = std::min(P{0}, x[i]);
    };

  asgard::moment_id melectric_x = pde.register_electric_moment(asgard::dimension_id(0), 2);
  asgard::moment_id melectric_y = pde.register_electric_moment(asgard::dimension_id(1), 2);

  auto md_positive_x = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> e_x = moments[melectric_x];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::max(P{0}, e_x[i]);
    };

  auto md_negative_x = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> e_x = moments[melectric_x];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::min(P{0}, e_x[i]);
    };

  auto md_positive_y = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> e_y = moments[melectric_y];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::max(P{0}, e_y[i]);
    };

  auto md_negative_y = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> e_y = moments[melectric_y];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::min(P{0}, e_y[i]);
    };

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
      asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
      asgard::term_identity{},
      asgard::term_volume<P>(positive),
      asgard::term_identity{}
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
    asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
    asgard::term_identity{},
    asgard::term_volume<P>(negative),
    asgard::term_identity{}
    });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
    asgard::term_identity{},
    asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
    asgard::term_identity{},
    asgard::term_volume<P>(positive)
  });

  pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
    asgard::term_identity{},
    asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
    asgard::term_identity{},
    asgard::term_volume<P>(negative)
    });

#ifdef ASGARD_USE_GPU
  auto gpu_md_positive_x = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_positive_kernel<<<blocks, threads>>>(num, f, moments[melectric_x].data(), fx);
  };

  auto gpu_md_negative_x = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_negative_kernel<<<blocks, threads>>>(num, f, moments[melectric_x].data(), fx);
  };

  auto gpu_md_positive_y = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_positive_kernel<<<blocks, threads>>>(num, f, moments[melectric_y].data(), fx);
  };

  auto gpu_md_negative_y = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_negative_kernel<<<blocks, threads>>>(num, f, moments[melectric_y].data(), fx);
  };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(gpu_md_positive_x, {melectric_x, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
        asgard::term_identity{}
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(gpu_md_negative_x, {melectric_x, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
        asgard::term_identity{}
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(gpu_md_positive_y, {melectric_y, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides))
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(gpu_md_negative_y, {melectric_y, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides))
      }
    };
#else
  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(md_positive_x, {melectric_x, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
        asgard::term_identity{}
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(md_negative_x, {melectric_x, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
        asgard::term_identity{}
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(md_positive_y, {melectric_y, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides))
      }
    };

  pde += asgard::term_md<P>{
      asgard::term_md(asgard::term_interp<P>(md_negative_y, {melectric_y, })),
      asgard::term_md<P>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides))
      }
    };
#endif

  // initial conditions in x and v
  auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 1.0 + 0.05 * std::cos(0.5 * x[i]);
    };

  auto ic_y = [](std::vector<P> const &y, P /* time */, std::vector<P> &fy) ->
    void {
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = 1.0 + 0.05 * std::cos(0.5 * y[i]);
    };

  auto ic_vx = [](std::vector<P> const &vx, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(2 * PI);

      for (size_t i = 0; i < vx.size(); i++)
        fv[i] = c * std::exp(-0.5 * vx[i] * vx[i]);
    };

  auto ic_vy = [](std::vector<P> const &vy, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(2 * PI);

      for (size_t i = 0; i < vy.size(); i++)
        fv[i] = c * std::exp(-0.5 * vy[i] * vy[i]);
    };

  pde.add_initial(asgard::separable_func<P>({ic_x, ic_y, ic_vx, ic_vy}));

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [landau make]
#endif
}

/*!
 * \ingroup asgard_examples_landau_damping
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_landau().
 *
 * \snippet landau_damping.cpp landau main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [landau main]
#endif

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 4D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the Landau damping Vlasov-Poisson in 2x-2v dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout << "<< additional options for this file >>\n";
    std::cout << "-test                               perform self-testing\n\n";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_landau(options),
                                         asgard::verbosity_level::low);

  // save the initial condition
  disc.add_aux_field({"initial condition", disc.current_state()});

  // save snapshots for every interval of time equal to 0.25
  // the stride is approximately the number of time-steps that make up 0.25
  int const stride = static_cast<int>(0.25 / disc.dt());

  // look over the entries and save multiple snapshots
  while (disc.remaining_steps() > 0
         and disc.advance_time(stride))
  {
    disc.progress_report();
    disc.add_aux_field({"snapshot time = " + std::to_string(disc.time()),
                        disc.current_state()});
  }

  // save final state
  disc.add_aux_field({"final state", disc.current_state()});

  // re-enable the output to show final stats
  disc.set_verbosity(asgard::verbosity_level::high);

  // write everything to a file
  disc.final_output();

  return 0;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [landau main]
#endif
};

#ifndef __ASGARD_DOXYGEN_SKIP
///////////////////////////////////////////////////////////////////////////////
// The code below is not part of the example, rather it is intended
// for correctness checking and verification against the known solution
///////////////////////////////////////////////////////////////////////////////

// just for convenience to avoid using asgard:: all over the place
// normally, should only include what is needed
using namespace asgard;

template<typename P>
void test_damping(std::string const &opt_str) {
  current_test<P> test_(opt_str, 2);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  // the pde needs only the zeroth moment and computes that internally
  // we are using the other moments to check energy conservation properties
  auto pde = make_landau(options);
  moment_id const melectric_x = pde.register_electric_moment(asgard::dimension_id(0), 2);
  moment_id const melectric_y = pde.register_electric_moment(asgard::dimension_id(1), 2);
  discretization_manager disc(std::move(pde), verbosity_level::quiet);

  int64_t const nt = disc.remaining_steps();
  std::vector<P> efield(nt);
  std::vector<P> time(nt);

  for (int64_t i = 0; i < nt; i++)
  {
    tassert( disc.advance_time(1) );

    disc.sync_mpi_state();

    if (not disc.has_poisson()) // in MPI context, do error checking only on Poisson-ranks
      continue;

    auto efieldx = disc.get_moment(melectric_x);
    auto efieldy = disc.get_moment(melectric_y);

    P Ep = 0;
    for (auto ex : efieldx) Ep += ex * ex;
    for (auto ey : efieldy) Ep += ey * ey;
    efield[i] = Ep;
    time[i] = disc.time();
  }
  std::vector<std::pair<P, P>> peaks;
  for (size_t i = 1; i < efield.size() - 1; i++) {
    if (efield[i] > efield[i - 1] and efield[i] > efield[i + 1])
      peaks.emplace_back(time[i], std::log(efield[i]));
  }

  P sum_t = 0;
  P sum_lnE = 0;
  P sum_t_lnE = 0;
  P sum_t2 = 0;
  size_t n = peaks.size();
  tassert(n > 1);
  for (std::pair<P, P> const &peak : peaks) {
    P t = peak.first;
    P lnE = peak.second;

    sum_t += t;
    sum_lnE += lnE;
    sum_t_lnE += t * lnE;
    sum_t2 += t * t;
  }
  P numerator = n * sum_t_lnE - sum_t * sum_lnE;
  P denominator = n * sum_t2 - sum_t * sum_t;
  P gamma = numerator / denominator;
  P gamma_theory = -0.3067189338;
  tassert(std::abs(gamma - gamma_theory) < 1.e-2);

  P omega_theory = 1.4156618886;
  for (size_t i = 1; i < peaks.size(); i++) {
    P dt = peaks[i].first - peaks[i - 1].first;
    P omega = asgard::PI / dt;
    tcheckless(i, std::abs(omega - omega_theory), 1.e-2);
  }
}

void self_test() {
  all_tests testing_("multi-d landau damping");

#ifdef ASGARD_ENABLE_DOUBLE

  test_damping<double>("-l 4 -a 1.e-3 -t 4.8");

#endif

#ifdef ASGARD_ENABLE_FLOAT
  current_test<float> test_("no-test, compile only");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
