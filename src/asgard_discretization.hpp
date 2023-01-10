#pragma once

#include "asgard_field.hpp"

namespace asgard
{
/*!
 * \internal
 * \ingroup AsgardPDE
 * \brief Wrapper around an adapt::distributed_grid but providing meta-data for the associated fields.
 *
 * Each asgard::field is associated with one field_discretization and each
 * discretization can be applied to multiple fields. This class holds an
 * instance of an adapt::distributed_grid and references to meta-data.
 *
 * \tparam precision is float or double
 * \tparam resrc indicates whether the wavelet transformer works on the host of GPU-device,
 *               currently supports only asgard::resource::host
 *
 * \endinternal
 */
template<typename precision, resource resrc>
struct field_discretization
{
  /*!
   * \brief Create a new discretization using the following parameters.
   *
   * \param cli_input a parser that overwrite some of the defaults
   * \param wavelet_transformer holds the cached values for the 1D transform
   * \param d_set is a the global set of all dimensions
   * \param d_name holds the subset of dimensions that will be used for the discretization
   *
   * \throws runtime_error if d_names contains names missing from the d_set
   */
  field_discretization(
      parser const &cli_input,
      asgard::basis::wavelet_transform<precision, resrc> &wavelet_transformer,
      dimension_set<precision> const &d_set,
      std::vector<std::string> const &d_names)
      : cli(cli_input), transformer(wavelet_transformer), state_size(0)
  {
    dims.reserve(d_names.size());
    for (size_t i = 0; i < d_names.size(); i++)
      dims.emplace_back(dimension<precision>(d_set(d_names[i])));

    grid =
        std::make_unique<adapt::distributed_grid<precision>>(cli_input, dims);
    auto const subgrid = grid->get_subgrid(get_rank());
    state_size         = (subgrid.col_stop - subgrid.col_start + 1) *
                 std::pow(dims.front().degree_, dims.size());
  }

  //! \brief Returns \b true if the dimensions of the field match the dimensions of the discretization grid.
  bool can_discretize(field<precision> const &field)
  {
    for (auto const &d_name : field.d_names)
    {
      if (not std::any_of(dims.begin(), dims.end(),
                          [&](dimension<precision> const &x) -> bool {
                            return (x.name == d_name);
                          }))
      {
        return false;
      }
    }
    return true;
  }

  //! \brief Returns the number of degrees of freedom for the discretization.
  int64_t size() const { return state_size; }

  /*!
   * \brief Writes the initial conditions associated with the field in the provided view.
   *
   * \param field is a field that is discretized by this grid
   * \param result is the output view where the write the data
   */
  void get_initial_conditions(field<precision> const &field,
                              fk::vector<precision, mem_type::view> result)
  {
    expect(result.size() == state_size);
    grid->get_initial_condition(cli, dims, field.init_cond, 1.0, transformer,
                                result);
  }
  //! \brief Overload that returns a copy of the initial condition vector.
  fk::vector<precision>
  get_initial_conditions(field_description<precision> const &field)
  {
    fk::vector<precision> result(state_size);
    get_initial_conditions(field,
                           fk::vector<precision, mem_type::view>(result));
    return result;
  }

  //! \brief Reference to the command line parser that overwrites some of the default parameters.
  parser const &cli;

  //! \brief References to the global cache for the 1D transform parameters
  asgard::basis::wavelet_transform<precision, resrc> &transformer;

  //! \brief The grid holding the hash-table.
  std::unique_ptr<adapt::distributed_grid<precision>> grid;
  //! \brief The number of degrees of freedom associated with the discretization.
  int64_t state_size;

  //! \brief The dimensions of the discretization.
  std::vector<dimension<precision>> dims;
};

} // namespace asgard
