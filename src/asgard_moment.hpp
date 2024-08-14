#pragma once
#include "asgard_basis.hpp"
#include "asgard_elements.hpp"
#include "asgard_transformations.hpp"

namespace asgard
{
#ifdef ASGARD_USE_CUDA
static constexpr resource sparse_resrc = resource::device;
#else
static constexpr resource sparse_resrc = resource::host;
#endif

template<typename P>
class moment
{
public:
  moment(std::vector<md_func_type<P>> md_funcs_);
  void createFlist(PDE<P> const &pde);
  void createMomentVector(PDE<P> const &pde,
                          elements::table const &hash_table);

  std::vector<md_func_type<P>> const &get_md_funcs() const { return md_funcs; }
  fk::vector<P> const &get_vector() const { return vector; }
  std::vector<std::vector<fk::vector<P>>> const &get_fList() const
  {
    return fList;
  }
  fk::matrix<P> const &get_moment_matrix() const { return moment_matrix; }
  fk::sparse<P, sparse_resrc> const &get_moment_matrix_dev() const
  {
    return sparse_mat;
  }

  void createMomentReducedMatrix(PDE<P> const &pde,
                                 elements::table const &hash_table);

  fk::vector<P> const &get_realspace_moment() const { return realspace; }
  void set_realspace_moment(fk::vector<P> &&realspace_in)
  {
    realspace = std::move(realspace_in);
  }

  fk::vector<P> &create_realspace_moment(
      PDE<P> const &pde_1d, fk::vector<P> &wave, elements::table const &table,
      asgard::basis::wavelet_transform<P, resource::host> const &transformer,
      std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace);

  fk::vector<P> &create_realspace_moment(
      PDE<P> const &pde_1d,
      fk::vector<P, mem_type::owner, resource::device> &wave,
      elements::table const &table,
      basis::wavelet_transform<P, resource::host> const &transformer,
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
  fk::sparse<P, sparse_resrc> sparse_mat;
};

} // namespace asgard
