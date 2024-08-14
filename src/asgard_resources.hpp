//-----------------------------------------------------------------------------
//
// device allocation and transfer helpers
//
//-----------------------------------------------------------------------------
#pragma once
#include "asgard_tools.hpp"

namespace asgard
{
// used to suppress warnings in unused variables
auto const ignore = [](auto ignored) { (void)ignored; };

/*!
 * \brief Default precision to use, double if enabled and float otherwise.
 */
#ifdef ASGARD_ENABLE_DOUBLE
using default_precision = double;
#else
using default_precision = float;
#endif

enum class resource
{
  host,
  device
};

/*!
 * \brief This it the type_identity template from C++-20.
 */
template<typename T>
struct type_identity_c17
{
  //! \brief Defines the same type as in the template parameter.
  using type = T;
};

//! \brief Blocks type deduction of templates, when it can cause issues.
template<typename T>
using no_deduce = typename type_identity_c17<T>::type;

/* tolerance for answer comparisons */
#define TOL std::numeric_limits<P>::epsilon() * 2

enum class mem_type
{
  owner,
  view,
  const_view
};

template<mem_type mem>
using enable_for_owner = std::enable_if_t<mem == mem_type::owner>;

template<mem_type mem>
using enable_for_all_views =
    std::enable_if_t<mem == mem_type::view || mem == mem_type::const_view>;

// enable only for const views
template<mem_type mem>
using enable_for_const_view = std::enable_if_t<mem == mem_type::const_view>;

// enable only for nonconst views
template<mem_type mem>
using enable_for_view = std::enable_if_t<mem == mem_type::view>;

// disable for const views
template<mem_type mem>
using disable_for_const_view =
    std::enable_if_t<mem == mem_type::owner || mem == mem_type::view>;

template<resource resrc>
using enable_for_host = std::enable_if_t<resrc == resource::host>;

template<resource resrc>
using enable_for_device = std::enable_if_t<resrc == resource::device>;

} // namespace asgard

namespace asgard::fk
{
template<typename P>
void allocate_device(P *&ptr, int64_t const num_elems,
                     bool const initialize = true);

template<typename P>
void delete_device(P *&ptr);

template<typename P>
void copy_on_device(P *const dest, P const *const source, int const num_elems);

template<typename P>
void copy_to_device(P *const dest, P const *const source, int const num_elems);

template<typename P>
void copy_to_host(P *const dest, P const *const source, int const num_elems);

template<resource resrc, typename P>
void allocate_resource(P *&ptr, int64_t const num_elems,
                       bool const initialize = true);

template<resource resrc, typename P>
void delete_resource(P *&ptr);

template<resource out, resource in, typename P>
void memcpy_1d(P *destination, P const *const source, int const num_elems);

template<resource out, resource in, typename P>
void memcpy_2d(P *dest, int const dest_stride, P const *const source,
               int const source_stride, int const nrows, int const ncols);

} // namespace asgard::fk

#ifdef ASGARD_USE_CUDA
#include "asgard_resources_cuda.tpp"
#else
#include "asgard_resources_host.tpp"
#endif
