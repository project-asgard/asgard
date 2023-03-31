#pragma once

namespace asgard
{
template<typename P>
class PDE;
}

#include "basis.hpp"
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

namespace basis
{
template<typename P, resource resrc>
class wavelet_transform;
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

  fk::vector<P> const &get_realspace_moment() const { return realspace; }

  fk::vector<P> &create_realspace_moment(
      PDE<P> const &pde_1d, fk::vector<P> &wave, elements::table const &table,
      asgard::basis::wavelet_transform<P, resource::host> const &transformer,
      std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace);

private:
  template<int nvdim>
  void createMomentReducedMatrix_nd(PDE<P> const &pde,
                                    elements::table const &hash_table);

  std::vector<md_func_type<P>> md_funcs;
  std::vector<std::vector<fk::vector<P>>> fList;
  fk::vector<P> vector;
  fk::matrix<P> moment_matrix;
  fk::vector<P> realspace;
  // moment_fval_integral;
  // moment_analytic_integral;
};

} // namespace asgard
