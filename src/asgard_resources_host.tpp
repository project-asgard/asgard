#include "asgard_tools.hpp"

namespace asgard::fk
{
template<typename P>
void allocate_device(P *&ptr, int64_t const num_elems, bool const initialize)
{
  ignore(ptr);
  ignore(num_elems);
  ignore(initialize);
  throw std::runtime_error("calling allocate_device without CUDA");
}

template<typename P>
void delete_device(P *&ptr)
{
  ignore(ptr);
  throw std::runtime_error("calling delete_device without CUDA");
}

template<typename P>
void copy_on_device(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_device without CUDA");
}

template<typename P>
void copy_to_device(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_device without CUDA");
}

template<typename P>
void copy_to_host(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_host without CUDA");
}

template<resource resrc, typename P>
void allocate_resource(P *&ptr, int64_t const num_elems, bool const initialize)
{
  static_assert(resrc == resource::host);

  if (num_elems == 0)
  {
    ptr = nullptr;
    return;
  }

  if (initialize)
    ptr = new P[num_elems]();
  else
    ptr = new P[num_elems];
}

template<resource resrc, typename P>
void delete_resource(P *&ptr)
{
  static_assert(resrc == resource::host);
  delete[] ptr;
  ptr = nullptr;
}

template<resource resrc, resource oresrc, typename P>
void memcpy_1d(P *dest, P const *const source, int const num_elems)
{
  static_assert(resrc == resource::host);
  static_assert(oresrc == resource::host);
  std::copy_n(source, num_elems, dest);
}

template<resource resrc, resource oresrc, typename P>
void memcpy_2d(P *dest, int const dest_stride, P const *const source,
               int const source_stride, int const nrows, int const ncols)
{
  static_assert(resrc == resource::host);
  static_assert(oresrc == resource::host);
#pragma omp parallel for
  for (int j = 0; j < ncols; ++j)
    std::copy_n(source + j * source_stride, nrows, dest + j * dest_stride);
}
} // namespace asgard::fk
