#pragma once

#include "pde/pde_base.hpp"
#include "program_options.hpp"
#include <vector>

template<typename P>
class moment
{
public:
  moment(std::vector<vector_func<P>> md_funcs_);
  void createFlist(PDE<P> const &pde, options const &opts);
  // createMomentVector()
private:
  std::vector<vector_func<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  // vector;
  // moment_fval_integral;
  // moment_analytic_integral;
};
