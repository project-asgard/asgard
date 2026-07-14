#include "asgard_test_macros.hpp"

using namespace asgard;

// solves u_xx = rhs over (xleft, xright), if bc is Dirichlet, dleft/dright are the boundary cond
// returns the result from comparison against the du_ref, which should be u_x
template<typename P>
P test_poisson_1d(std::function<P(P)> du_ref, std::function<P(P)> rhs, P xleft, P xright,
               P dleft, P dright, poisson_bc const bc, int degree, int level)
{
  poisson_1d<P> solver(degree, xleft, xright, level, moment_id{0}, moment_id{1});

  // construct the cell-by-cell Legendre expansion of the rhs
  // we must switch to std::vector functions
  auto lrhs = [&](std::vector<P> const &x, std::vector<P> &fx)
      -> void {
          for (auto i : indexof(x))
            fx[i] = - rhs(x[i]);
      };
  auto rref = [&](std::vector<P> const &x, std::vector<P> &fx)
      -> void {
          // the solver computes the gevative-gradient
          for (auto i : indexof(x))
            fx[i] = - du_ref(x[i]);
      };

  // the hierarchy manipulatro can do the projection
  hierarchy_manipulator<P> hier(degree, 1, {xleft, }, {xright, });

  int constexpr dim0 = 0;
  std::vector<P> vrhs = hier.cell_project(dim0, lrhs, level);
  std::vector<P> sv; // will hold the output

  solver.solve(vrhs, dleft, dright, bc, sv);

  // the output sv holds the cell-by-cell constant values of the gradient
  // comput reference expansion of the provided reference gradient
  hierarchy_manipulator<P> hier0(0, 1, {xleft, }, {xright, });

  std::vector<P> vref = hier0.cell_project(dim0, rref, level);

  // vref is the pw-constant expansion of rref over the non-hierarchical cells
  // the Legenre polynomials are scaled to unit norm, to get the point-wise values
  // we must rescale back
  P const scale = std::sqrt(fm::ipow2(level) / (xright - xleft));
  for (auto &v : vref)
    v *= scale;

  return fm::diff_inf(sv, vref);
}

template<typename P>
struct PoissonErrors
{
  P phi;
  P ex;
  P ey;
};

// solves u_xx = rhs over (xleft, xright), if bc is Dirichlet, dleft/dright are the boundary cond
// returns the result from comparison against the du_ref, which should be u_x
template<typename P>
PoissonErrors<P> test_poisson_md(separable_func<P> rhs, std::array<P, 2> xleft, std::array<P, 2> xright,
                                 separable_func<P> phi, separable_func<P> ex, separable_func<P> ey,
                                 int degree, int level)
{
  prog_opts options;
  options.degree = degree;
  options.start_levels = {level, level};
  pde_domain<P> domain(position_dims{2}, velocity_dims{0},
                       {{xleft[0], xright[0]}, {xleft[1], xright[1]}});
  pde_scheme<P> pde(options, domain);
  pde.set_initial([](P, vector2d<P>, std::vector<P>){}); // triggers has_interp = true
  term_manager<P> terms = term_manager<P>(options, domain, pde, sparse_grid(options));

  // setup poisson solver
  moments_list mlist;
  moment_id mex = mlist.get_add_id(moment::electric(dimension_id{0}, 2));
  moment_id mey = mlist.get_add_id(moment::electric(dimension_id{1}, 2));
  auto build_func = [&terms](term_entry<P> &tentry, int const dim, int const lvl) -> void {
    terms.rebuild_term1d(tentry, dim, lvl);
  };
  poisson_md<P> poisson(2, level, terms.xleft, terms.xright, terms.conn, terms.hier, mlist, build_func, moment_id{0});
  poisson.update_preconditioner(terms.grid, terms.conn, poisson_bc::periodic);

  // fill vectors from functions
  int n = terms.grid.num_dof();
  std::vector<P> rhs_ref(n);
  std::vector<P> phi_ref(n);
  std::vector<P> ex_ref(n);
  std::vector<P> ey_ref(n);
  terms.hier.project_separable(rhs, terms.grid, {}, P{0}, P{1}, rhs_ref.data());
  terms.hier.project_separable(phi, terms.grid, {}, P{0}, P{1}, phi_ref.data());
  terms.hier.project_separable(ex, terms.grid, {}, P{0}, P{1}, ex_ref.data());
  terms.hier.project_separable(ey, terms.grid, {}, P{0}, P{1}, ey_ref.data());

  // solve
  momentset<P> moms(2);
  #ifdef ASGARD_USE_GPU
  compute->set_device(gpu::device{0});
  gpu::vector<P> gpu_rhs_ref(n);
  gpu_rhs_ref.copy_from_host(n, rhs_ref.data());
  auto interp_func = [&](gpu::vector<P> const &efield, moment_id mid) -> void {
    efield.copy_to_host(moms[mid]);
  };
  poisson.solve_periodic(gpu_rhs_ref, terms.grid, terms.conn, interp_func, terms.kwork);
  gpu::vector<P> const &gpu_phi_wav = poisson.get_potential();
  std::vector<P> phi_wav(n);
  gpu_phi_wav.copy_to_host(phi_wav);
  #else
  poisson.solve_periodic(rhs_ref, moms, terms.grid, terms.conn, terms.kwork);
  std::vector<P> const &phi_wav = poisson.get_potential();
  #endif
  std::vector<P> const &ex_wav = moms[mex];
  std::vector<P> const &ey_wav = moms[mey];

  return PoissonErrors<P> ({diff_l2(n, phi_wav.data(), phi_ref.data()),
                            diff_l2(n, ex_wav.data(), ex_ref.data()),
                            diff_l2(n, ey_wav.data(), ey_ref.data())});
}

template<typename TestType>
void poisson_tests()
{
  TestType tol = (std::is_same_v<TestType, double>) ? 1.E-14 : 1.E-5;

  {
    current_test<TestType> name_("poisson - const-gradient, low degree");
    int const degree = 0;
    int const level  = 3;

    // example 1, u = x over (-2, 3), du = 1, ddu = 0
    auto rhs = [](TestType)->TestType { return TestType{0}; };
    auto du  = [](TestType)->TestType { return TestType{1}; };

    TestType err = test_poisson_1d<TestType>(
        du, rhs, -2, 3, -2, 3, poisson_bc::dirichlet, degree, level);

    tassert(err < tol);
  }
  {
    current_test<TestType> name_("poisson - const-gradient, high degree");
    int const degree = 2;
    int const level  = 5;

    // example 1, using higher degree and level
    auto rhs = [](TestType)->TestType { return TestType{0}; };
    auto du  = [](TestType)->TestType { return TestType{1}; };

    TestType err = test_poisson_1d<TestType>(
        du, rhs, -2, 3, -2, 3, poisson_bc::dirichlet, degree, level);

    tassert(err < tol);
  }
  {
    current_test<TestType> name_("poisson - variable-gradient");
    int const degree = 1;
    int const level  = 4;

    // example 1, u = x over (-2, 3), du = 1, ddu = 0
    auto rhs = [](TestType)->TestType { return TestType{2}; };
    auto du  = [](TestType x)->TestType { return TestType{2} * x; };

    TestType err = test_poisson_1d<TestType>(
        du, rhs, -2, 3, 4, 9, poisson_bc::dirichlet, degree, level);

    tassert(err < tol);
  }
  {
    current_test<TestType> name_("poisson - messy-gradient");
    // do not attempt this in single precision
    if (std::is_same_v<TestType, float>)
      return;

    TestType constexpr pi = 3.141592653589793;

    int const degree = 2;
    int const level  = 9;

    // example 2, u = sin(pi * x) over (-1, 1), du = pi * cos(pi * x),
    //            ddu = -pi^2 * sin(pi * x), ddu = 0
    auto rhs = [](TestType x)->TestType { return -pi * pi * std::sin(pi * x) - 1; };
    auto du  = [](TestType x)->TestType { return pi * std::cos(pi * x); };

    TestType err = test_poisson_1d<TestType>(
        du, rhs, -1, 1, 5, 11, poisson_bc::periodic, degree, level);

    tassert(err < 1.E-8);
  }

  TestType poisson_md_tol = (std::is_same_v<TestType, double>) ? 1.E-6 : 1.E-4;

  {
    current_test<TestType> name_("poisson md - uniform-y");

    TestType constexpr pi = 3.141592653589793;

    int const degree = 3;
    int const level = 8;

    // example 1, rhs(x, y) = 1 - 0.5 * cos(pi * x) over x: (-1, 1) and y: (-1, 1),
    // phi(x, y) = 0.5 * x^2 + 0.5 * cos(pi * x)/pi^2 + 0.5 * (1/pi^2 - 1)
    // E_x = x + 0.5 * sin(pi * x)/pi, E_y = 0
    auto rhs_x = [](TestType x)->TestType { return 1 - 0.5 * std::cos(pi * x); };
    auto phi_x = [](TestType x)->TestType { return -0.5 * std::cos(pi * x) / (pi * pi); };
    auto ex_x = [](TestType x)->TestType { return -0.5 * std::sin(pi * x) / pi; };
    auto zero = [](TestType)->TestType { return 0; };
    auto one = [](TestType)->TestType { return 1; };
    separable_func<TestType> rhs{{vectorize<TestType>(rhs_x), vectorize<TestType>(one)}};
    separable_func<TestType> phi{{vectorize<TestType>(phi_x), vectorize<TestType>(one)}};
    separable_func<TestType> ex{{vectorize<TestType>(ex_x), vectorize<TestType>(one)}};
    separable_func<TestType> ey{{vectorize<TestType>(zero), vectorize<TestType>(zero)}};

    PoissonErrors<TestType> errs = test_poisson_md<TestType>(
        rhs, {-1, -1}, {1, 1}, phi, ex, ey, degree, level);

    tassert(errs.phi < poisson_md_tol);
    tassert(errs.ex < poisson_md_tol);
    tassert(errs.ey < poisson_md_tol);
  }
  {
    current_test<TestType> name_("poisson md - manufactured");

    TestType constexpr pi = 3.141592653589793;

    int const degree = 3;
    int const level = 8;

    // example 2, rhs(x, y) =  4 * exp(-x^2 - y^2) * (x^2 + y^2 - 1) over x: (-5, 5) and y: (-5, 5),
    // phi(x, y) = exp(-x^2 - y^2)
    // E_x = -2x * exp(-x^2 - y^2), E_y = -2y * exp(-x^2 - y^2)
    auto rhs_x = [](TestType x)->TestType { return -2 * std::sin(x); };
    auto rhs_y = [](TestType y)->TestType { return std::sin(y); };
    auto phi_x = [](TestType x)->TestType { return -std::sin(x); };
    auto phi_y = [](TestType y)->TestType { return std::sin(y); };
    auto ex_x = [](TestType x)->TestType { return std::cos(x); };
    auto ex_y = [](TestType y)->TestType { return std::sin(y); };
    auto ey_x = [](TestType x)->TestType { return std::sin(x); };
    auto ey_y = [](TestType y)->TestType { return std::cos(y); };
    separable_func<TestType> rhs{{vectorize<TestType>(rhs_x), vectorize<TestType>(rhs_y)}};
    separable_func<TestType> phi{{vectorize<TestType>(phi_x), vectorize<TestType>(phi_y)}};
    separable_func<TestType> ex{{vectorize<TestType>(ex_x), vectorize<TestType>(ex_y)}};
    separable_func<TestType> ey{{vectorize<TestType>(ey_x), vectorize<TestType>(ey_y)}};

    PoissonErrors<TestType> errs = test_poisson_md<TestType>(
        rhs, {0, 0}, {2*pi, 2*pi}, phi, ex, ey, degree, level);

    tassert(errs.phi < poisson_md_tol);
    tassert(errs.ex < poisson_md_tol);
    tassert(errs.ey < poisson_md_tol);
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("poisson tests", " builtin poisson solver functionality");

  #ifdef ASGARD_ENABLE_DOUBLE
  poisson_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  poisson_tests<float>();
  #endif

  return 0;
}
