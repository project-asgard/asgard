#include "asgard_resources.hpp"
#include "build_info.hpp"

#include <algorithm>
#include <stdexcept>

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
  for (int j = 0; j < ncols; ++j)
    std::copy_n(source + j * source_stride, nrows, dest + j * dest_stride);
}

// TODO #ifdef leads to linking errors
//#ifdef ASGARD_ENABLE_DOUBLE
template void
allocate_device(double *&ptr, int64_t const num_elems, bool const initialize);

template void delete_device(double *&ptr);

template void copy_on_device(double *const dest, double const *const source,
                             int const num_elems);

template void copy_to_device(double *const dest, double const *const source,
                             int const num_elems);

template void copy_to_host(double *const dest, double const *const source,
                           int const num_elems);

template void allocate_resource<resource::host>(double *&ptr,
                                                int64_t const num_elems,
                                                bool const initialize);

template void delete_resource<resource::host>(double *&ptr);

template void
memcpy_1d<resource::host, resource::host>(double *dest,
                                          double const *const source,
                                          int const num_elems);

template void
memcpy_2d<resource::host, resource::host>(double *dest, int const dest_stride,
                                          double const *const source,
                                          int const source_stride,
                                          int const nrows, int const ncols);

//#endif

//#ifdef ASGARD_ENABLE_FLOAT
template void
allocate_device(float *&ptr, int64_t const num_elems, bool const initialize);

template void delete_device(float *&ptr);

template void copy_on_device(float *const dest, float const *const source,
                             int const num_elems);

template void copy_to_device(float *const dest, float const *const source,
                             int const num_elems);

template void allocate_resource<resource::host>(float *&ptr,
                                                int64_t const num_elems,
                                                bool const initialize);

template void delete_resource<resource::host>(float *&ptr);

template void
copy_to_host(float *const dest, float const *const source, int const num_elems);

template void
memcpy_1d<resource::host, resource::host>(float *dest,
                                          float const *const source,
                                          int const num_elems);

template void
memcpy_2d<resource::host, resource::host>(float *dest, int const dest_stride,
                                          float const *const source,
                                          int const source_stride,
                                          int const nrows, int const ncols);

//#endif

template void
allocate_device(int *&ptr, int64_t const num_elems, bool const initialize);

template void delete_device(int *&ptr);

template void
copy_on_device(int *const dest, int const *const source, int const num_elems);

template void
copy_to_device(int *const dest, int const *const source, int const num_elems);

template void
copy_to_host(int *const dest, int const *const source, int const num_elems);

template void allocate_resource<resource::host>(int *&ptr,
                                                int64_t const num_elems,
                                                bool const initialize);

template void delete_resource<resource::host>(int *&ptr);

template void memcpy_1d<resource::host, resource::host>(int *dest,
                                                        int const *const source,
                                                        int const num_elems);

template void
memcpy_2d<resource::host, resource::host>(int *dest, int const dest_stride,
                                          int const *const source,
                                          int const source_stride,
                                          int const nrows, int const ncols);
} // namespace asgard::fk
