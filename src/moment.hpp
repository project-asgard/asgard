#pragma once
#include "elements.hpp"
#include "pde/pde_base.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <vector>

template<typename P>
class moment
{
public:
  moment(std::vector<vector_func<P>> md_funcs_);
  void createFlist(PDE<P> const &pde, options const &opts);
  void
  createMomentVector(parser const &opts, elements::table const &hash_table);

private:
  std::vector<vector_func<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  fk::vector<P> vector;
  // moment_fval_integral;
  // moment_analytic_integral;
};
