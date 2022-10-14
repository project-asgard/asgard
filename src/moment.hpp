#pragma once

template<typename P>
class PDE;

#include "elements.hpp"
#include "pde/pde_base.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <vector>

template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const, P const)>;

namespace elements
{
class table;
}

template<typename P>
class moment
{
public:
  moment(std::vector<vector_func<P>> md_funcs_);
  void createFlist(PDE<P> const &pde, options const &opts);
  void
  createMomentVector(parser const &opts, elements::table const &hash_table);

  std::vector<vector_func<P>> const &get_md_funcs() const { return md_funcs; }
  fk::matrix<P> const &get_moment_matrix() const { return moment_matrix; }

  void createMomentReducedMatrix(PDE<P> const &pde, options const &opts,
                                 elements::table const &hash_table,
                                 int const moment_idx);

private:
  std::vector<vector_func<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  fk::vector<P> vector;
  fk::matrix<P> moment_matrix;
  // moment_fval_integral;
  // moment_analytic_integral;
};
