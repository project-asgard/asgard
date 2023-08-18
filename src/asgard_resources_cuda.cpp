#include "asgard_resources.hpp"
#include "build_info.hpp"
#include "tools.hpp"

#include <cuda_runtime.h>

namespace asgard::fk
{
template<typename P>
void allocate_device(P *&ptr, int64_t const num_elems, bool const initialize)
{
  allocate_resource<resource::device>(ptr, num_elems, initialize);
}

template<typename P>
void delete_device(P *&ptr)
{
  delete_resource<resource::device>(ptr);
}

template<typename P>
void copy_on_device(P *const dest, P const *const source, int const num_elems)
{
  memcpy_1d<resource::device, resource::device>(dest, source, num_elems);
}

template<typename P>
void copy_to_device(P *const dest, P const *const source, int const num_elems)
{
  memcpy_1d<resource::device, resource::host>(dest, source, num_elems);
}

template<typename P>
void copy_to_host(P *const dest, P const *const source, int const num_elems)
{
  memcpy_1d<resource::host, resource::device>(dest, source, num_elems);
}

template<resource resrc, typename P>
void allocate_resource(P *&ptr, int64_t const num_elems, bool const initialize)
{
  if constexpr (resrc == resource::host)
  {
    if (initialize)
      ptr = new P[num_elems]();
    else
      ptr = new P[num_elems];
  }
  else if constexpr (resrc == resource::device)
  {
    if (cudaMalloc((void **)&ptr, num_elems * sizeof(P)) != cudaSuccess)
      throw std::bad_alloc();
    if (num_elems > 0)
      expect(ptr != nullptr);
    if (initialize)
    {
      auto success = cudaMemset((void *)ptr, 0, num_elems * sizeof(P));
      expect(success == cudaSuccess);
    }
  }
}

template<resource resrc, typename P>
void delete_resource(P *&ptr)
{
  if constexpr (resrc == resource::host)
  {
    delete[] ptr;
  }
  else if constexpr (resrc == resource::device)
  {
    auto const success = cudaFree(ptr);
    // the device runtime may be unloaded at process shut down
    // (when static storage duration destructors are called)
    // returning a cudartUnloading error code.
    expect((success == cudaSuccess) || (success == cudaErrorCudartUnloading));
  }
  ptr = nullptr;
}

static constexpr cudaMemcpyKind
getCudaMemcpyKind(resource destination, resource source)
{
  if (destination == resource::host)
  {
    if (source == resource::host)
      return cudaMemcpyHostToHost;
    else if (source == resource::device)
      return cudaMemcpyDeviceToHost;
  }
  else if (destination == resource::device)
  {
    if (source == resource::host)
      return cudaMemcpyHostToDevice;
    else if (source == resource::device)
      return cudaMemcpyDeviceToDevice;
  }
}

template<resource out, resource in, typename P>
void memcpy_1d(P *dest, P const *const source, int const num_elems)
{
  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(out, in);
  auto const success = cudaMemcpy(dest, source, num_elems * sizeof(P), kind);
  expect(success == cudaSuccess);
}

template<resource out, resource in, typename P>
void memcpy_2d(P *dest, int const dest_stride, P const *const source,
               int const source_stride, int const nrows, int const ncols)
{
  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(out, in);

  auto const success =
      cudaMemcpy2D(dest, dest_stride * sizeof(P), source,
                   source_stride * sizeof(P), nrows * sizeof(P), ncols, kind);
  expect(success == cudaSuccess);
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
template void allocate_resource<resource::device>(double *&ptr,
                                                  int64_t const num_elems,
                                                  bool const initialize);
template void delete_resource<resource::host>(double *&ptr);
template void delete_resource<resource::device>(double *&ptr);

template void
memcpy_1d<resource::device, resource::device>(double *dest,
                                              double const *const source,
                                              int const num_elems);
template void
memcpy_1d<resource::device, resource::host>(double *dest,
                                            double const *const source,
                                            int const num_elems);
template void
memcpy_1d<resource::host, resource::device>(double *dest,
                                            double const *const source,
                                            int const num_elems);
template void
memcpy_1d<resource::host, resource::host>(double *dest,
                                          double const *const source,
                                          int const num_elems);

template void memcpy_2d<resource::device, resource::device>(
    double *dest, int const dest_stride, double const *const source,
    int const source_stride, int const nrows, int const ncols);
template void
memcpy_2d<resource::device, resource::host>(double *dest, int const dest_stride,
                                            double const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
template void
memcpy_2d<resource::host, resource::device>(double *dest, int const dest_stride,
                                            double const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
template void
memcpy_2d<resource::host, resource::host>(double *dest, int const dest_stride,
                                          double const *const source,
                                          int const source_stride,
                                          int const nrows, int const ncols);
//#endif

// TODO #ifdef leads to linking errors
//#ifdef ASGARD_ENABLE_FLOAT
template void
allocate_device(float *&ptr, int64_t const num_elems, bool const initialize);

template void delete_device(float *&ptr);

template void copy_on_device(float *const dest, float const *const source,
                             int const num_elems);

template void copy_to_device(float *const dest, float const *const source,
                             int const num_elems);

template void
copy_to_host(float *const dest, float const *const source, int const num_elems);

template void allocate_resource<resource::host>(float *&ptr,
                                                int64_t const num_elems,
                                                bool const initialize);
template void allocate_resource<resource::device>(float *&ptr,
                                                  int64_t const num_elems,
                                                  bool const initialize);
template void delete_resource<resource::host>(float *&ptr);
template void delete_resource<resource::device>(float *&ptr);

template void
memcpy_1d<resource::device, resource::device>(float *dest,
                                              float const *const source,
                                              int const num_elems);
template void
memcpy_1d<resource::device, resource::host>(float *dest,
                                            float const *const source,
                                            int const num_elems);
template void
memcpy_1d<resource::host, resource::device>(float *dest,
                                            float const *const source,
                                            int const num_elems);
template void
memcpy_1d<resource::host, resource::host>(float *dest,
                                          float const *const source,
                                          int const num_elems);

template void memcpy_2d<resource::device, resource::device>(
    float *dest, int const dest_stride, float const *const source,
    int const source_stride, int const nrows, int const ncols);
template void
memcpy_2d<resource::device, resource::host>(float *dest, int const dest_stride,
                                            float const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
template void
memcpy_2d<resource::host, resource::device>(float *dest, int const dest_stride,
                                            float const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
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
template void allocate_resource<resource::device>(int *&ptr,
                                                  int64_t const num_elems,
                                                  bool const initialize);
template void delete_resource<resource::host>(int *&ptr);
template void delete_resource<resource::device>(int *&ptr);

template void
memcpy_1d<resource::device, resource::device>(int *dest,
                                              int const *const source,
                                              int const num_elems);
template void
memcpy_1d<resource::device, resource::host>(int *dest, int const *const source,
                                            int const num_elems);
template void
memcpy_1d<resource::host, resource::device>(int *dest, int const *const source,
                                            int const num_elems);
template void memcpy_1d<resource::host, resource::host>(int *dest,
                                                        int const *const source,
                                                        int const num_elems);

template void
memcpy_2d<resource::device, resource::device>(int *dest, int const dest_stride,
                                              int const *const source,
                                              int const source_stride,
                                              int const nrows, int const ncols);
template void
memcpy_2d<resource::device, resource::host>(int *dest, int const dest_stride,
                                            int const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
template void
memcpy_2d<resource::host, resource::device>(int *dest, int const dest_stride,
                                            int const *const source,
                                            int const source_stride,
                                            int const nrows, int const ncols);
template void
memcpy_2d<resource::host, resource::host>(int *dest, int const dest_stride,
                                          int const *const source,
                                          int const source_stride,
                                          int const nrows, int const ncols);

} // namespace asgard::fk
