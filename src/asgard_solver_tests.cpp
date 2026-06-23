#include "asgard_test_macros.hpp"

using namespace asgard;

// tests the consistency between solver_method enum order and the order
// inside the solver_manager.var variant
template<int idx, solver_method method, typename P, typename solver_type>
void match_solver() {
  tassert(idx == static_cast<int>(method));
  solver_manager<P> solver;
  static_assert(std::is_same_v<decltype(std::get<idx>(solver.var)), solver_type&>,
                "inconsistent index for the solver_manager::var");
}

template<typename P>
void solver_manager_tests()
{
  current_test<P> name_("solver manager");

  match_solver<0, solver_method::direct, P, solvers::direct<P>>();
  match_solver<1, solver_method::bicgstab, P, solvers::bicgstab<P>>();
  match_solver<2, solver_method::gmres, P, solvers::gmres<P>>();
  match_solver<3, solver_method::scaled_identity, P, solvers::scaled_identity<P>>();
}

int main(int argc, char **argv) {

  libasgard_runtime running_(argc, argv);

  all_tests global_("solver tests", " builtin solver functionality");

  #ifdef ASGARD_ENABLE_DOUBLE
  solver_manager_tests<double>();
  #endif

  #ifdef ASGARD_ENABLE_FLOAT
  solver_manager_tests<float>();
  #endif

  return 0;
}
