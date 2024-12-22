#include "asgard.hpp"

using namespace asgard;

using prec = asgard::default_precision;

template<typename P>
P test_poisson(std::function<P(P)> du_ref, std::function<P(P)> rhs, P xleft, P xright,
               P dleft, P dright, solver::poisson_bc const bc, int degree, int level)
{
  solver::poisson_data<P> solver(degree, xleft, xright, level);

  // construct the cell-by-cell Legenre expansion of the rhs
  // we must switch to std::vector functions
  auto lrhs = [&](std::vector<P> const &x, std::vector<P> &fx)
      -> void {
          for (auto i : indexof(x))
            fx[i] = rhs(x[i]);
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

int main(int argc, char **argv)
{
  ignore(argc);
  ignore(argv);
  // keep this file clean for each PR
  // allows someone to easily come here, dump code and start playing
  // this is good for prototyping and quick-testing features/behavior

  using TestType = double;

// do not attempt this in single precision

    TestType constexpr pi = 3.141592653589793;

    int const degree = 2;
    int const level  = 3;

    // example 2, u = sin(pi * x) over (-1, 1), du = pi * cos(pi * x),
    //            ddu = -pi^2 * sin(pi * x), ddu = 0
    auto rhs = [](TestType x)->TestType { return -pi * pi * std::sin(pi * x) - 1; };
    auto du  = [](TestType x)->TestType { return -pi * std::cos(pi * x); };

    TestType err = test_poisson<TestType>(
        du, rhs, -1, 1, 5, 11, solver::poisson_bc::periodic, degree, level);

    std::cout << " error = " << err << "\n";

  return 0;
}
