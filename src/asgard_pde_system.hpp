#pragma once

#include "asgard_discretization.hpp"

/*!
 * \defgroup AsgardPDESystem Asgard PDE system
 *
 * \par PDE System
 * Asgard can solve a system of partial differential equations (PDE)
 * described by multiple dimensions, fields and equations.
 * The data associated with the system is held in an object of the
 * asgard::pde_system class.
 *
 * \par Components
 * The PDE system is comprised by several components provided by the user.
 * - command line input
 * - a list (std::vector) of asgard::dimension_description objects
 * - a list (std::vector) of asgard::field_description objects
 */

namespace asgard
{
/*!
 * \internal
 * \ingroup AsgardPDESystem
 * \brief Converts the user-provided descriptions into field objects that are used internally in the pde_system.
 *
 * Used in the initialization (constructor) of the pde_system objects.
 * \endinternal
 */
template<typename precision>
std::vector<field<precision>> make_fields_vector(
    std::vector<field_description<precision>> const &descriptions)
{
  std::vector<field<precision>> result;
  result.reserve(descriptions.size());
  for (auto const &d : descriptions)
    result.push_back(field<precision>(d));
  return result;
}

/*!
 * \ingroup AsgardPDESystem
 * \brief Object of this class will contain the data describing a PDE system.
 *
 * \tparam precision is either \b float or \b double which indicates the use of 32-bit or 64-bit precision
 * \tparam rescr only supports asgard::resource::host, eventually will support asgard::resource::device for GPU support
 *
 * \par
 * The pde_system is object is the main container for the data associated with
 * PDE system. Time-stepping and solvers operate on the pde_system.
 *
 * Internally, the object will also store all relevant information about the
 * sparse grid discretization (e.g., the hash-tables) and will also hold the
 * current PDE state (i.e., the solution at the current time).
 *
 * \internal
 * \par
 * The constructor takes the user-provided descriptions objects,
 * but will convert those into the internal asgard::dimension and asgard::field
 * objects. The pde_system also contains the table associated with the
 * discretization. \b note : wll also contain the operators. \endinternal
 */
template<typename precision, resource resrc = asgard::resource::host>
class pde_system
{
public:
  /*!
   * \brief Constructor that initializes the internal data-structures.
   *
   * \param cli_input contains the command line inputs that can overwrite the Asgard defaults for discretization order and density
   * \param dimensions contains the descriptions for all dimensions associated with the pde-system
   * \param flieds_terms contains the descriptions for the fields used by the PDE system
   *
   * \throws runtime_error if the fields or dimensions have entries with repeated names
   *         or if the fields are associated with non-existing dimensions
   *
   * \par
   * Each filed must be associated with dimensions from the provided list.
   * The fields and dimensions are related by the "name" parameters.
   */
  pde_system(parser const &cli_input,
             std::vector<dimension_description<precision>> const &dimensions,
             std::vector<field_description<precision>> const &field_terms)
      : cli(cli_input), dims(cli, dimensions),
        fields(make_fields_vector(field_terms)),
        transformer(cli_input, dimensions.front().degree, true)
  {
    static_assert(std::is_same<precision, float>::value or
                      std::is_same<precision, double>::value,
                  "ASGARD supports only float and double as template "
                  "parameters for precision.");

    expect(dimensions.size() > 0);

    for (auto &f : field_terms)
      f.verify_dimensions(dims);

    std::vector<std::string> field_names(fields.size());
    for (size_t i = 0; i < field_names.size(); i++)
      field_names[i] = fields[i].name;

    if (not check_unique_strings(field_names))
      throw std::runtime_error(
          "pde-system created with repeated fields (same names)");

    // eventually we will give the user a more fine-grained control to the
    // discretization this method is the default, i.e., use default
    // discretization currently, we only allow default discretization
    finalizes_discretization();
  }

  /*!
   * \internal
   * \brief Finishes the initialization and constructs the discretization grid objects.
   *
   * This is automatically called by the constructor and
   * also computes the relative offsets of each field win the global vector of
   * degrees of freedom.
   *
   * TODO: move to protected
   * \endinternal
   */
  void finalizes_discretization()
  {
    for (size_t i = 0; i < fields.size(); i++)
    {
      // for each field, see if we already have discretization for this field
      int64_t grid_index = 0;
      while (grid_index < static_cast<int64_t>(grids.size()))
      {
        if (grids[grid_index].can_discretize(fields[i]))
        {
          fields[i].grid_index = grid_index;
          break;
        }
        grid_index++;
      }
      if (grid_index == static_cast<int64_t>(grids.size()))
      {
        grids.push_back(field_discretization<precision, resrc>(
            cli, transformer, dims, fields[i].d_names));
        fields[i].grid_index = grid_index;
      }
    }

    int64_t total_size = 0;
    for (auto &f : fields)
    {
      int64_t size = grids[f.grid_index].size();
      f.set_global_index(total_size, total_size + size);
      total_size += size;
    }

    state.resize(total_size);
  }

  //! \brief Overwrites the current state of the PDE System with initial conditions provided by the fields.
  void load_initial_conditions()
  {
    for (auto const &f : fields)
    {
      grids[f.grid_index].get_initial_conditions(
          f, fk::vector<precision, mem_type::view>(state, f.global_begin,
                                                   f.global_end - 1));
    }
  }

  /*!
   * \brief Return a const-reference to the entries associated with the given field name.
   *
   * \param name is the name of the requested dimension.
   *
   * \returns a const-reference to the sub-vector associated with the current field.
   *
   * \throws runtime_error if the \b name is not a valid field name.
   */
  fk::vector<precision, mem_type::const_view> get_field(std::string const &name)
  {
    size_t findex = 0;
    while (findex < fields.size() && fields[findex].name != name)
      findex++;

    auto f = std::find_if(fields.cbegin(), fields.cend(),
                          [&](field<precision> const &candidate) -> bool {
                            return (candidate.name == name);
                          });

    if (f == fields.end())
      throw std::runtime_error("field name '" + name + "' is not unrecognized");

    return fk::vector<precision, mem_type::const_view>(state, f->global_begin,
                                                       f->global_end - 1);
  }

  // TODO: will work with operators similar to field
  // there is not need to have a separate "field_set",
  // we can just cross-reference the names of fields in the operator with the
  // names in the fields vector
  void add_operator() {}

private:
  parser const &cli;
  dimension_set<precision> dims;
  std::vector<field<precision>> fields;

  asgard::basis::wavelet_transform<precision, resrc> transformer;
  std::vector<field_discretization<precision, resrc>> grids;

  fk::vector<precision> state;
};

} // namespace asgard
