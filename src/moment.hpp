#pragma once

namespace asgard
{
template<typename P>
class PDE;
}

#include "elements.hpp"
#include "pde/pde_base.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include <vector>

namespace asgard
{
namespace elements
{
class table;
}

template<typename P>
class moment
{
public:
  moment(std::vector<md_func_type<P>> md_funcs_);
  void createFlist(PDE<P> const &pde, options const &opts);
  void createMomentVector(PDE<P> const &pde, parser const &opts,
                          elements::table const &hash_table);

  std::vector<md_func_type<P>> const &get_md_funcs() const { return md_funcs; }
  fk::vector<P> const &get_vector() const { return vector; }
  std::vector<std::vector<fk::vector<P>>> const &get_fList() const
  {
    return fList;
  }
  fk::matrix<P> const &get_moment_matrix() const { return moment_matrix; }

  void createMomentReducedMatrix(PDE<P> const &pde,
                                 elements::table const &hash_table);

private:
  std::vector<md_func_type<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  fk::vector<P> vector;
  fk::matrix<P> moment_matrix;
  // moment_fval_integral;
  // moment_analytic_integral;
};

} // namespace asgard
