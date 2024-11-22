#pragma once
#include "asgard_pde_base.hpp"

namespace asgard
{
// ---------------------------------------------------------------------------
//
// the "continuity 6d" pde
//
// 6D test case using continuity equation, i.e.,
// df/dt + b.grad_x(f) + a.grad_v(f)==0 where b={1,1,3}, a={4,3,2}
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_continuity_6d : public PDE<P>
{
public:
  PDE_continuity_6d(prog_opts const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields used to check correctness of specification
  static int constexpr num_dims_           = 6;
  static int constexpr num_sources_        = 7;
  static int constexpr num_terms_          = 6;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // define some reusable functions
  static fk::vector<P> f0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    return fk::vector<P>(std::vector<P>(x.size(), 0.0));
  }

  // specify initial condition vector functions...

  // specify exact solution vectors/time function...

  static P constexpr targ  = 2;
  static P constexpr xarg  = M_PI;
  static P constexpr yarg  = M_PI / 2;
  static P constexpr zarg  = M_PI / 3;
  static P constexpr vxarg = M_PI / 10;
  static P constexpr vyarg = M_PI / 20;
  static P constexpr vzarg = M_PI / 30;

  // pde.analytic_solutions_1D = { ...
  //    @(x,p,t) cos(xarg*x), ...
  //    @(y,p,t) sin(yarg*y), ...
  //    @(z,p,t) cos(zarg*z), ...
  //    @(vx,p,t) cos(vxarg*vx), ...
  //    @(vy,p,t) sin(vyarg*vy), ...
  //    @(vz,p,t) cos(vzarg*vz), ...
  //    @(t)   sin(targ*t)
  //    };

  static fk::vector<P> exact_solution_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> exact_solution_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }

  static fk::vector<P> exact_solution_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> exact_solution_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> exact_solution_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }

  static fk::vector<P> exact_solution_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }

  static fk::vector<P> exact_time(fk::vector<P>, P const time)
  {
    return {
        std::sin(targ * time),
    };
  }

  // define exact soln
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_x, exact_solution_y, exact_solution_z,
      exact_solution_vx, exact_solution_vy, exact_solution_vz, exact_time};

  // specify source functions...

  // source 0
  // source0 = { ...
  //    @(x,p)  cos(xarg*x), ...
  //    @(y,p)  sin(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) cos(vxarg*vx), ...
  //    @(vy,p) sin(vyarg*vy), ...
  //    @(vz,p) cos(vzarg*vz), ...
  //    @(t)  2*cos(targ*t)    ...
  //    };
  static fk::vector<P> source_0_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_0_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_0_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_0_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_0_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_0_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_0_time(P const time) { return 2.0 * std::cos(targ * time); }
  inline static source<P> const source0_ =
      source<P>({source_0_x, source_0_y, source_0_z, source_0_vx, source_0_vy,
                 source_0_vz},
                source_0_time);

  // source 1
  // source1 = { ...
  //    @(x,p)  cos(xarg*x), ...
  //    @(y,p)  cos(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) cos(vxarg*vx), ...
  //    @(vy,p) sin(vyarg*vy), ...
  //    @(vz,p) cos(vzarg*vz), ...
  //    @(t)  1/2*pi*sin(targ*t)    ...
  //    };

  static fk::vector<P> source_1_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_1_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_1_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_1_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_1_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_1_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_1_time(P const time)
  {
    return 0.5 * PI * std::sin(targ * time);
  }
  inline static source<P> const source1_ =
      source<P>({source_1_x, source_1_y, source_1_z, source_1_vx, source_1_vy,
                 source_1_vz},
                source_1_time);

  // source 2
  // source2 = { ...
  //    @(x,p)  sin(xarg*x), ...
  //    @(y,p)  sin(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) cos(vxarg*vx), ...
  //    @(vy,p) sin(vyarg*vy), ...
  //    @(vz,p) cos(vzarg*vz), ...
  //    @(t)  -pi*sin(targ*t)    ...
  //    };
  static fk::vector<P> source_2_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_2_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_2_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_2_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_2_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_2_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_2_time(P const time) { return -PI * std::sin(targ * time); }
  inline static source<P> const source2_ =
      source<P>({source_2_x, source_2_y, source_2_z, source_2_vx, source_2_vy,
                 source_2_vz},
                source_2_time);

  // source 3
  // source3 = { ...
  //   @(x,p)  cos(xarg*x), ...
  //   @(y,p)  sin(yarg*y), ...
  //   @(z,p)  sin(zarg*z), ...
  //   @(vx,p) cos(vxarg*vx), ...
  //   @(vy,p) sin(vyarg*vy), ...
  //   @(vz,p) cos(vzarg*vz), ...
  //   @(t)  -pi*sin(targ*t)    ...
  //   };

  static fk::vector<P> source_3_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_3_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_3_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_3_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_3_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_3_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_3_time(P const time) { return -PI * std::sin(targ * time); }
  inline static source<P> const source3_ =
      source<P>({source_3_x, source_3_y, source_3_z, source_3_vx, source_3_vy,
                 source_3_vz},
                source_3_time);

  // source 4
  // source4 = { ...
  //    @(x,p)  cos(xarg*x), ...
  //    @(y,p)  sin(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) cos(vxarg*vx), ...
  //    @(vy,p) cos(vyarg*vy), ...
  //    @(vz,p) cos(vzarg*vz), ...
  //    @(t)  3/20*pi*sin(targ*t)    ...
  //    };

  static fk::vector<P> source_4_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_4_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_4_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_4_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_4_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_4_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_4_time(P const time)
  {
    return 3.0 / 20.0 * PI * std::sin(targ * time);
  }
  inline static source<P> const source4_ =
      source<P>({source_4_x, source_4_y, source_4_z, source_4_vx, source_4_vy,
                 source_4_vz},
                source_4_time);

  // source 5
  // source5 = { ...
  //    @(x,p)  cos(xarg*x), ...
  //    @(y,p)  sin(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) sin(vxarg*vx), ...
  //    @(vy,p) sin(vyarg*vy), ...
  //    @(vz,p) cos(vzarg*vz), ...
  //    @(t)  -2/5*pi*sin(targ*t)    ...
  //    };
  static fk::vector<P> source_5_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_5_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_5_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_5_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_5_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_5_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vzarg * x_v); });
    return fx;
  }
  static P source_5_time(P const time)
  {
    return -2.0 / 5.0 * PI * std::sin(targ * time);
  }
  inline static source<P> const source5_ =
      source<P>({source_5_x, source_5_y, source_5_z, source_5_vx, source_5_vy,
                 source_5_vz},
                source_5_time);

  // source 6
  // source6 = { ...
  //    @(x,p)  cos(xarg*x), ...
  //    @(y,p)  sin(yarg*y), ...
  //    @(z,p)  cos(zarg*z), ...
  //    @(vx,p) cos(vxarg*vx), ...
  //    @(vy,p) sin(vyarg*vy), ...
  //    @(vz,p) sin(vzarg*vz), ...
  //    @(t)  -1/15*pi*sin(targ*t)    ...
  //    };

  static fk::vector<P> source_6_x(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(xarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_6_y(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(yarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_6_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(zarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_6_vx(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::cos(vxarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_6_vy(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vyarg * x_v); });
    return fx;
  }
  static fk::vector<P> source_6_vz(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x_v) { return std::sin(vzarg * x_v); });
    return fx;
  }
  static P source_6_time(P const time)
  {
    return -1.0 / 15.0 * PI * std::sin(targ * time);
  }
  inline static source<P> const source6_ =
      source<P>({source_6_x, source_6_y, source_6_z, source_6_vx, source_6_vy,
                 source_6_vz},
                source_6_time);

  // define sources list

  inline static std::vector<source<P>> const sources_ = {
      source0_, source1_, source2_, source3_, source4_, source5_, source6_};

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dt      = x_range / fm::ipow2(dim.get_level());
    return dt;
  }

  // g-funcs for terms (optional)

  static P gx(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -bx;
  }
  static P gy(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -by;
  }
  static P gz(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -bz;
  }
  static P gvx(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -ax;
  }
  static P gvy(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -ay;
  }
  static P gvz(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -az;
  }

  // define dimensions
  inline static dimension<P> const x_ = dimension<P>(-1.0, // domain min
                                                     1.0,  // domain max
                                                     2,    // levels
                                                     1,    // degree
                                                     f0,   // initial condition
                                                     nullptr,
                                                     "x"); // name

  inline static dimension<P> const y_ = dimension<P>(-2.0, // domain min
                                                     2.0,  // domain max
                                                     2,    // levels
                                                     1,    // degree
                                                     f0,   // initial condition
                                                     nullptr,
                                                     "y"); // name

  inline static dimension<P> const z_ = dimension<P>(-3.0, // domain min
                                                     3.0,  // domain max
                                                     2,    // levels
                                                     1,    // degree
                                                     f0,   // initial condition
                                                     nullptr,
                                                     "z"); // name

  inline static dimension<P> const vx_ = dimension<P>(-10.0, // domain min
                                                      10.0,  // domain max
                                                      2,     // levels
                                                      1,     // degree
                                                      f0, // initial condition
                                                      nullptr,
                                                      "vx"); // name

  inline static dimension<P> const vy_ = dimension<P>(-20.0, // domain min
                                                      20.0,  // domain max
                                                      2,     // levels
                                                      1,     // degree
                                                      f0, // initial condition
                                                      nullptr,
                                                      "vy"); // name

  inline static dimension<P> const vz_ = dimension<P>(-30.0, // domain min
                                                      30.0,  // domain max
                                                      2,     // levels
                                                      1,     // degree
                                                      f0, // initial condition
                                                      nullptr,
                                                      "z"); // name

  inline static std::vector<dimension<P>> const dimensions_ = {x_,  y_,  z_,
                                                               vx_, vy_, vz_};

  // define terms
  static P constexpr bx = 1;
  static P constexpr by = 1;
  static P constexpr bz = 3;
  static P constexpr ax = 4;
  static P constexpr ay = 3;
  static P constexpr az = 2;

  // default mass matrix (only for lev_x=lev_y=etc)
  inline static partial_term<P> const partial_term_I_ = partial_term<P>(
      coefficient_type::mass, nullptr, nullptr, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);
  inline static term<P> const I_ = term<P>(false,   // time-dependent
                                           "massY", // name
                                           {partial_term_I_});

  // term 0
  inline static partial_term<P> const partial_term_0_x_ = partial_term<P>(
      coefficient_type::div, gx, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term0_x_ = term<P>(false,      // time-dependent
                                                 "v_x.d_dx", // name
                                                 {partial_term_0_x_});
  inline static std::vector<term<P>> const terms0_ = {term0_x_, I_, I_,
                                                      I_,       I_, I_};
  // term 1
  inline static partial_term<P> const partial_term_1_y_ = partial_term<P>(
      coefficient_type::div, gy, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term1_y_ = term<P>(false,      // time-dependent
                                                 "v_y.d_dy", // name
                                                 {partial_term_1_y_});
  inline static std::vector<term<P>> const terms1_ = {I_, term1_y_, I_,
                                                      I_, I_,       I_};
  // term 2
  inline static partial_term<P> const partial_term_2_z_ = partial_term<P>(
      coefficient_type::div, gz, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term2_z_ = term<P>(false,      // time-dependent
                                                 "v_z.d_dz", // name
                                                 {partial_term_2_z_});

  inline static std::vector<term<P>> const terms2_ = {I_, I_, term2_z_,
                                                      I_, I_, I_};
  // term 3
  inline static partial_term<P> const partial_term_3_vx_ = partial_term<P>(
      coefficient_type::div, gvx, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term3_vx_ = term<P>(false,       // time-dependent
                                                  "a_x.d_dvx", // name
                                                  {partial_term_3_vx_});

  inline static std::vector<term<P>> const terms3_ = {I_,        I_, I_,
                                                      term3_vx_, I_, I_};
  // term 4
  inline static partial_term<P> const partial_term_4_vy_ = partial_term<P>(
      coefficient_type::div, gvy, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term4_vy_ = term<P>(false,       // time-dependent
                                                  "a_y.d_dvy", // name
                                                  {partial_term_4_vy_});

  inline static std::vector<term<P>> const terms4_ = {I_, I_,        I_,
                                                      I_, term4_vy_, I_};
  // term 5
  inline static partial_term<P> const partial_term_5_vz_ = partial_term<P>(
      coefficient_type::div, gvz, nullptr, flux_type::upwind,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const term5_vz_ = term<P>(false,       // time-dependent
                                                  "a_z.d_dvz", // name
                                                  {partial_term_5_vz_});

  inline static std::vector<term<P>> const terms5_ = {I_, I_, I_,
                                                      I_, I_, term5_vz_};

  /* this corresponds to the aggregation of TERM_ND objects in matlab */
  inline static term_set<P> const terms_ = {terms0_, terms1_, terms2_,
                                            terms3_, terms4_, terms5_};
};
} // namespace asgard
