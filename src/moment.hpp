#pragma once

#include "pde/pde_base.hpp"
#include <vector>

template<typename P>
class moment
{
public:
  moment(std::vector<vector_func<P>> md_func);
  // createFlist();
  // createMomentVector()
private:
  std::vector<vector_func<P>> md_func_;
  // fList;
  // vector;
  // moment_fval_integral;
  // moment_analytic_integral;
};
