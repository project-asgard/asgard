//-----------------------------------------------------------------------------
//
// device allocation and transfer helpers
//
//-----------------------------------------------------------------------------
#pragma once

#include "lib_dispatch.hpp"

#include <cstdint>

namespace asgard
{
// used to suppress warnings in unused variables
auto const ignore = [](auto ignored) { (void)ignored; };
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
