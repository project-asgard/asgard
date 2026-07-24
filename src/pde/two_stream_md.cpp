#include "asgard.hpp"

#include "asgard_test_macros.hpp" // only for testing

/*!
 * \internal
 * \file two_stream.cpp
 * \brief Two stream instability example
 * \author The ASGarD Team
 * \ingroup asgard_examples_two_stream
 *
 * \endinternal
 */

/*!
 * \ingroup asgard_examples
 * \addtogroup asgard_examples_two_stream Example: Two stream instability
 *
 * \par Two stream instability
 * Solves the Vlasov-Poisson equation in a common example
 * often called the two stream instability problem
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
 * \ingroup asgard_examples_two_stream
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
  while (i < num) {
    // Avoiding std::max to prevent __device__ compilation header conflicts
    out[i] = field[i] * ((mom[i] > 0.0) ? mom[i] : 0.0); 
    i += blockDim.x * gridDim.x;
  }
}

template<typename P>
__global__ void interp_negative_kernel(int64_t num, P const* field, P const* mom, P* out) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num) {
    out[i] = field[i] * ((mom[i] < 0.0) ? mom[i] : 0.0);
    i += blockDim.x * gridDim.x;
  }
}

template<typename P>
__global__ void weight_kernel_2d(int64_t num, P const *field, P const *ex, P const *ey, P* out) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num) {
    out[i] = field[i] * (ex[i] * ex[i] + ey[i] * ey[i]);
    i += blockDim.x * gridDim.x;
  }
}

template<typename P>
__global__ void weight_kernel_3d(int64_t num, P const *field, P const *ex, P const *ey, P const *ez, P* out) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num) {
    out[i] = field[i] * (ex[i] * ex[i] + ey[i] * ey[i] + ez[i] * ez[i]);
    i += blockDim.x * gridDim.x;
  }
}
#endif

/*!
 * \ingroup asgard_examples_two_stream
 * \brief Make single two-stream PDE
 *
 * Constructs the pde description for the given umber of dimensions
 * and options.
 *
 * \tparam P is either double or float, the asgard::default_precision will select
 *           first double, if unavailable, will go for float
 *
 * \param options is the set of options
 *
 * \returns the asgard::pde_scheme definition
 *
 * \snippet two_stream.cpp two_stream make
 */
template<typename P = asgard::default_precision>
asgard::pde_scheme<P> make_two_stream(int const pos_dims, asgard::prog_opts options) {
#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream make]
#endif

  options.title = std::to_string(pos_dims) + "X" + std::to_string(pos_dims) + "V Two Stream Instability";

  asgard::pde_domain<P> domain;
  if (pos_dims == 2) {
    domain = asgard::pde_domain<P>(asgard::position_dims{2}, asgard::velocity_dims{2},
                                 {{-2 * PI, 2 * PI}, {-2 * PI, 2 * PI},
                                  {-2 * PI, 2 * PI}, {-2 * PI, 2 * PI}});
    options.default_start_levels = {7, 7, 7, 7};
  } else if (pos_dims == 3) {
    domain = asgard::pde_domain<P>(asgard::position_dims{3}, asgard::velocity_dims{3},
                                 {{-2 * PI, 2 * PI}, {-2 * PI, 2 * PI},
                                  {-2 * PI, 2 * PI}, {-2 * PI, 2 * PI},
                                  {-2 * PI, 2 * PI}, {-2 * PI, 2 * PI}});
    options.default_start_levels = {6, 6, 6, 6, 6, 6};
  } else {
    throw std::runtime_error("dims must be 2 or 3");
  }

  // setting some default options
  // defaults are used only the corresponding values are missing from the command line
  int const default_degree = 2;

  options.default_degree = default_degree;
  options.default_poisson_tolerance = 1e-8;
  options.default_poisson_iterations = 1000;
  options.default_poisson_precon = asgard::precon_method::jacobi;

  options.default_plotter_colormap = "viridis";

  // the CFL is more complicated, it depends both on the polynomial degree
  // and on the maximum number of cells (TODO: add more here)
  int const k = options.degree.value_or(default_degree);
  int const n = (1 << options.max_level());
  options.default_dt = 3.0 / (2 * (2 * k + 1) * n);

  options.default_stop_time = 2.0;

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

  asgard::moment_id melectric_x;
  asgard::moment_id melectric_y;
  asgard::moment_id melectric_z;
  if (pos_dims == 2) {
    melectric_x = pde.register_electric_moment(asgard::dimension_id(0), 2);
    melectric_y = pde.register_electric_moment(asgard::dimension_id(1), 2);
  } else {
    melectric_x = pde.register_electric_moment(asgard::dimension_id(0), 3);
    melectric_y = pde.register_electric_moment(asgard::dimension_id(1), 3);
    melectric_z = pde.register_electric_moment(asgard::dimension_id(2), 3);
  }

  auto md_positive_x = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_x = moments[melectric_x];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::max(P{0}, e_x[i]);
    };

  auto md_negative_x = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_x = moments[melectric_x];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::min(P{0}, e_x[i]);
    };

  auto md_positive_y = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_y = moments[melectric_y];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::max(P{0}, e_y[i]);
    };

  auto md_negative_y = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_y = moments[melectric_y];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::min(P{0}, e_y[i]);
    };

  auto md_positive_z = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_z = moments[melectric_z];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::max(P{0}, e_z[i]);
    };

  auto md_negative_z = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                    asgard::momentset<P> const &moments, std::vector<P> const &field,
                    std::vector<P> &vals)
    {
      std::vector<P> const &e_z = moments[melectric_z];
#pragma omp parallel for
      for (size_t i = 0; i < vals.size(); i++)
        vals[i] = field[i] * std::min(P{0}, e_z[i]);
    };


  if (pos_dims == 2) {
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
  } else {
    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(positive),
        asgard::term_identity{},
        asgard::term_identity{}
      });

    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(negative),
        asgard::term_identity{},
        asgard::term_identity{}
      });

    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_identity{},
        asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(positive),
        asgard::term_identity{}
      });

    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_identity{},
        asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(negative),
        asgard::term_identity{}
      });

    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(positive)
      });

    pde += asgard::term_md<P>(std::vector<asgard::term_1d<P>>{
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::periodic),
        asgard::term_identity{},
        asgard::term_identity{},
        asgard::term_volume<P>(negative)
      });
  }

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

  auto gpu_md_positive_z = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_positive_kernel<<<blocks, threads>>>(num, f, moments[melectric_z].data(), fx);
  };

  auto gpu_md_negative_z = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[]) {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;
      interp_negative_kernel<<<blocks, threads>>>(num, f, moments[melectric_z].data(), fx);
  };

  std::function<void(int64_t, P, P const[], asgard::momentset_gpu<P> const&, P const[], P[])> weight;

  if (pos_dims == 2) {
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

    weight = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[])
    {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;

      asgard::gpu::vector<P> const &e_x = moments[melectric_x];
      asgard::gpu::vector<P> const &e_y = moments[melectric_y];
      weight_kernel_2d<<<blocks, threads>>>(num, f, e_x.data(), e_y.data(), fx);
    };

  } else {
    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_positive_x, {melectric_x, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{},
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_negative_x, {melectric_x, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{},
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_positive_y, {melectric_y, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_negative_y, {melectric_y, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{}
        }
      };

      pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_positive_z, {melectric_z, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides))
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(gpu_md_negative_z, {melectric_z, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides))
        }
      };

    weight = [=](int64_t num, P t, P const x[], asgard::momentset_gpu<P> const &moments, P const f[], P fx[])
    {
      int threads = 256;
      int blocks = (num + threads - 1) / threads;

      asgard::gpu::vector<P> const &e_x = moments[melectric_x];
      asgard::gpu::vector<P> const &e_y = moments[melectric_y];
      asgard::gpu::vector<P> const &e_z = moments[melectric_z];
      weight_kernel_3d<<<blocks, threads>>>(num, f, e_x.data(), e_y.data(), e_z.data(), fx);
    };
  }

#else
  std::function<void(P, asgard::vector2d<P> const&, asgard::momentset<P> const&,
                     std::vector<P> const&, std::vector<P>&)> weight;

  if (pos_dims == 2) {
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

    weight = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                      asgard::momentset<P> const &moments, std::vector<P> const &field,
                      std::vector<P> &vals)
      {
        std::vector<P> const &e_x = moments[melectric_x];
        std::vector<P> const &e_y = moments[melectric_y];
  #pragma omp parallel for
        for (size_t i = 0; i < vals.size(); i++)
          vals[i] = field[i] * (e_x[i] * e_x[i] + e_y[i] * e_y[i]);
      };
  } else {
    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_positive_x, {melectric_x, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{},
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_negative_x, {melectric_x, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{},
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_positive_y, {melectric_y, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_negative_y, {melectric_y, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides)),
          asgard::term_identity{}
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_positive_z, {melectric_z, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::upwind, asgard::boundary_type::bothsides))
        }
      };

    pde += asgard::term_md<P>{
        asgard::term_md(asgard::term_interp<P>(md_negative_z, {melectric_z, })),
        asgard::term_md<P>{
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_identity{},
          asgard::term_1d<P>(asgard::term_div<P>(1, asgard::flux_type::downwind, asgard::boundary_type::bothsides))
        }
      };

    weight = [=](P /* time */, asgard::vector2d<P> const& /* nodes */,
                      asgard::momentset<P> const &moments, std::vector<P> const &field,
                      std::vector<P> &vals)
      {
        std::vector<P> const &e_x = moments[melectric_x];
        std::vector<P> const &e_y = moments[melectric_y];
        std::vector<P> const &e_z = moments[melectric_z];
  #pragma omp parallel for
        for (size_t i = 0; i < vals.size(); i++)
          vals[i] = field[i] * (e_x[i] * e_x[i] + e_y[i] * e_y[i] + e_z[i] * e_z[i]);
      };
  }
#endif

  // initial conditions in x and v
  auto ic_x = [](std::vector<P> const &x, P /* time */, std::vector<P> &fx) ->
    void {
      for (size_t i = 0; i < x.size(); i++)
        fx[i] = 1.0 - 0.5 * std::cos(0.5 * x[i]);
    };

  auto ic_y = [](std::vector<P> const &y, P /* time */, std::vector<P> &fy) ->
    void {
      for (size_t i = 0; i < y.size(); i++)
        fy[i] = 1.0;
    };

  auto ic_z = [](std::vector<P> const &z, P /* time */, std::vector<P> &fz) ->
    void {
      for (size_t i = 0; i < z.size(); i++)
        fz[i] = 1.0;
    };

  auto ic_vx = [](std::vector<P> const &vx, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{2} / std::sqrt(PI);

      for (size_t i = 0; i < vx.size(); i++)
        fv[i] = c * vx[i] * vx[i] * std::exp(-vx[i] * vx[i]);
    };

  auto ic_vy = [](std::vector<P> const &vy, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(PI);

      for (size_t i = 0; i < vy.size(); i++)
        fv[i] = c * std::exp(-vy[i] * vy[i]);
    };

  auto ic_vz = [](std::vector<P> const &vz, P /* time */, std::vector<P> &fv) ->
    void {
      P const c = P{1} / std::sqrt(PI);

      for (size_t i = 0; i < vz.size(); i++)
        fv[i] = c * std::exp(-vz[i] * vz[i]);
    };

  if (pos_dims == 2) {
    pde.add_initial(asgard::separable_func<P>({ic_x, ic_y, ic_vx, ic_vy}));
    pde.set_adapt_weight(weight, {melectric_x, melectric_y});
  } else {
    pde.add_initial(asgard::separable_func<P>({ic_x, ic_y, ic_z, ic_vx, ic_vy, ic_vz}));
    pde.set_adapt_weight(weight, {melectric_x, melectric_y, melectric_z});
  }

  return pde;

#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream make]
#endif
}

/*!
 * \ingroup asgard_examples_two_stream
 * \brief main() for the diffusion example
 *
 * The main() processes the command line arguments and calls make_two_stream().
 *
 * \snippet two_stream.cpp two_stream main
 */
int main(int argc, char** argv)
{
#ifndef __ASGARD_DOXYGEN_SKIP
//! [two_stream main]
#endif

  // if MPI is enabled, call MPI_Init(), otherwise do nothing
  asgard::libasgard_runtime running_(argc, argv);

  // if double precision is available the P is double
  // otherwise P is float
  using P = asgard::default_precision;

  // parse the command-line inputs
  asgard::prog_opts options(argc, argv);

  // if help was selected in the command line, show general information about
  // this example runs 2D problem, testing does more options
  if (options.show_help) {
    std::cout << "\n solves the two stream Vlasov-Poisson in 2x-2v dimensions\n\n";
    std::cout << "    -- standard ASGarD options --";
    options.print_help(std::cout);
    std::cout <<
R"help(<< additional options for this file >>
-dims            -dm     int        accepts: 2 or 3
                                    the number of position dimensions
-test                               perform self-testing
)help";
    return 0;
  }

  // this is an optional step, check if there are misspelled or incorrect cli entries
  // the first set/vector of entries are those that can appear by themselves
  // the second set/vector requires extra parameters
  options.throw_if_argv_not_in({"-test", "--test"}, {"-dims", "-dm"});

  if (options.has_cli_entry("-test") or options.has_cli_entry("--test")) {
    // perform series of internal tests, not part of the example/tutorial
    self_test();
    return 0;
  }

  int const pos_dims = options.extra_cli_value_group<int>({"-dims", "-dm"}).value_or(2);

  // the discretization_manager takes in a pde and handles sparse-grid construction
  // separable and non-separable operators, holds the current state, etc.
  asgard::discretization_manager<P> disc(make_two_stream(pos_dims, options),
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
//! [two_stream main]
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
void test_energy(std::string const &opt_str) {
  current_test<P> test_(opt_str, 2);
  // analytic solution is not available, hence we use energy conservation for
  // the test quantity in place of an L^2 error

  prog_opts const options = make_opts(opt_str);

  // the pde needs only the zeroth moment and computes that internally
  // we are using the other moments to check energy conservation properties
  auto pde = make_two_stream(2, options);
  moment_id const rho = pde.register_moment({0, 0});
  moment_id const p0 = pde.register_moment({1, 0}); // needed for verification, but not running
  moment_id const p1 = pde.register_moment({0, 1});
  moment_id const ke0 = pde.register_moment({2, 0});
  moment_id const ke1 = pde.register_moment({0, 2});
  moment_id const melectric_x = pde.register_electric_moment(asgard::dimension_id(0), 2);
  moment_id const melectric_y = pde.register_electric_moment(asgard::dimension_id(1), 2);
  discretization_manager disc(std::move(pde), verbosity_level::quiet);

  P E0 = 0; // initial total energy (potential + kinetic), will initialize on first iteration

  int64_t const n = disc.remaining_steps();

  for (int64_t i = 0; i < n; i++)
  {
    tassert( disc.advance_time(1) );

    disc.sync_mpi_state();

    if (not disc.has_poisson()) // in MPI context, do error checking only on Poisson-ranks
      continue;

    P area = disc.domain().length(0) * disc.domain().length(1);

    auto efieldx = disc.get_moment(melectric_x);
    auto efieldy = disc.get_moment(melectric_y);

    P Ep = 0;
    for (auto ex : efieldx) Ep += ex * ex;
    for (auto ey : efieldy) Ep += ey * ey;

    std::vector<P> momke0 = disc.get_moment(ke0);
    std::vector<P> momke1 = disc.get_moment(ke1);
    P Ek = (momke0[0] + momke1[0]) * std::sqrt(area);

    if (disc.current_step() == 1) // first time-step
      E0 = 0.5 * (Ep + Ek);

    // std::cout << "Total energy error: " << std::abs(0.5 * (Ep + Ek) - E0) << "\n";
    tcheckless(i, std::abs(0.5 * (Ep + Ek) - E0), 7.E-6);

    std::vector<P> mom0 = disc.get_moment(rho);
    std::vector<P> momp0 = disc.get_moment(p0);
    std::vector<P> momp1 = disc.get_moment(p1);

    // integral of moment 0 by moment 1, by delta_ij orthogonality of the basis
    // just sum up the product of the coefficients
    P mv0 = 0;
    for (size_t j = 0; j < mom0.size(); j++)
      mv0 += mom0[j] * momp0[j];
    P mv1 = 0;
    for (size_t j = 0; j < mom0.size(); j++)
      mv1 += mom0[j] * momp1[j];

    // std::cout << "X momentum error: " << mv0 << "\n";
    // std::cout << "Y momentum error: " << mv1 << "\n\n";
    tcheckless(i, std::abs(mv0), 2.0e-5);
    tcheckless(i, std::abs(mv1), 2.0e-5);

    // check the initial slight energy decay before it stabilizes
    if (i > 0)
      tassert(std::abs(Ep + Ek - E0) > 1.E-9);
  }
}

void self_test() {
  all_tests testing_("multi-d two-stream instability");

#ifdef ASGARD_ENABLE_DOUBLE

  test_energy<double>("-l 6 -d 3 -n 10 -dt 6.25e-3 -a 1.0e-6 -ppc none");
  test_energy<double>("-s rk4 -l 6 -d 2 -n 5 -dt 6.25e-3 -a 1.0e-6");

#endif

#ifdef ASGARD_ENABLE_FLOAT
  current_test<float> test_("no-test, compile only");
#endif
}

#endif //__ASGARD_DOXYGEN_SKIP
