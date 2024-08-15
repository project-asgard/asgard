#pragma once
#include "asgard_transformations.hpp"

namespace asgard
{
template<typename P>
void generate_all_coefficients(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);

template<typename P>
void generate_all_coefficients_max_level(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0.0, bool const rotate = true);

template<typename P>
void generate_dimension_mass_mat(
    PDE<P> &pde,
    basis::wavelet_transform<P, resource::host> const &transformer);

template<typename P>
fk::matrix<P> generate_coefficients(
    dimension<P> const &dim, partial_term<P> const &pterm,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const level, P const time = 0.0, bool const rotate = true);

/*!
 * \internal
 * \brief Stores matrices associated with a pde
 *
 * Stores the matrices of a PDE discretization. Starting with the mass matrices,
 * which are stored in block-diagonal format.
 *
 * \endinternal
 */
template<typename P>
class coefficient_matrix_manager
{
public:
  coefficient_matrix_manager(PDE<P> *pde)
      : pde_(pde), num_dims_(pde_->num_dims())
  {
    for (int d : indexof<int>(num_dims_))
      if (pde_->get_dimensions()[d].volume_jacobian_dV)
      {
        mass_[d] = std::make_unique<mass_matrix<P>>();
        if (dv_funcs_.empty())
          dv_funcs_.resize(num_dims_, nullptr); // nullptr means default to 1
        dv_funcs_[d] = [=](std::vector<P> const &x, std::vector<P> &fx)
            -> void {
          for (auto i : indexof(x))
            fx[i] = pde_->get_dimensions()[d].volume_jacobian_dV(x[i], 0);
        };
      }
  }

  auto &mass() const { return mass_; }
  auto const &dv() const { return dv_funcs_; }

private:
  PDE<P> *pde_ = nullptr;

  int num_dims_;
  mutable std::array<std::unique_ptr<mass_matrix<P>>, max_num_dimensions> mass_;
  std::vector<function_1d<P>> dv_funcs_;
};

} // namespace asgard
