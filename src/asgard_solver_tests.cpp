#include "asgard_test_macros.hpp"

using namespace asgard;

// solves u_xx = rhs over (xleft, xright), if bc is Dirichlet, dleft/dright are the boundary cond
// returns the result from comparison against the du_ref, which should be u_x
template<typename P>
P test_poisson(std::function<P(P)> du_ref, std::function<P(P)> rhs, P xleft, P xright,
               P dleft, P dright, solvers::poisson_bc const bc, int degree, int level)
{
  solvers::poisson<P> solver(degree, xleft, xright, level);

  // construct the cell-by-cell Legenre expansion of the rhs
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

  std::vector<P> vrhs = hier.cell_project(lrhs, nullptr, level);
  std::vector<P> sv; // will hold the output

  solver.solve(vrhs, dleft, dright, bc, sv);

  // the output sv holds the cell-by-cell constant values of the gradient
  // comput reference expansion of the provided reference gradient
  hierarchy_manipulator<P> hier0(0, 1, {xleft, }, {xright, });

  std::vector<P> vref = hier0.cell_project(rref, nullptr, level);

  // vref is the pw-constant expansion of rref over the non-hierarchical cells
  // the Legenre polynomials are scaled to unit norm, to get the point-wise values
  // we must rescale back
  P const scale = std::sqrt(fm::ipow2(level) / (xright - xleft));
  for (auto &v : vref)
    v *= scale;

  return fm::diff_inf(sv, vref);
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

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, -2, 3, solvers::poisson_bc::dirichlet, degree, level);

    tassert(err < tol);
  }
  {
    current_test<TestType> name_("poisson - const-gradient, high degree");
    int const degree = 2;
    int const level  = 5;

    // example 1, using higher degree and level
    auto rhs = [](TestType)->TestType { return TestType{0}; };
    auto du  = [](TestType)->TestType { return TestType{1}; };

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, -2, 3, solvers::poisson_bc::dirichlet, degree, level);

    tassert(err < tol);
  }
  {
    current_test<TestType> name_("poisson - variable-gradient");
    int const degree = 1;
    int const level  = 4;

    // example 1, u = x over (-2, 3), du = 1, ddu = 0
    auto rhs = [](TestType)->TestType { return TestType{2}; };
    auto du  = [](TestType x)->TestType { return TestType{2} * x; };

    TestType err = test_poisson<TestType>(
        du, rhs, -2, 3, 4, 9, solvers::poisson_bc::dirichlet, degree, level);

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

    TestType err = test_poisson<TestType>(
        du, rhs, -1, 1, 5, 11, solvers::poisson_bc::periodic, degree, level);

    tassert(err < 1.E-8);
  }
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("solver tests", " builtin solver functionality");

  #ifdef ASGARD_ENABLE_DOUBLE
  poisson_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  poisson_tests<float>();
  #endif

  return 0;
}
