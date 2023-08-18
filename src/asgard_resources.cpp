#include "asgard_resources.hpp"

namespace asgard::fk
{
#ifdef ASGARD_USE_CUDA
template<typename P>
void allocate_device(P *&ptr, int64_t const num_elems, bool const initialize)
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
#else
template<typename P>
void allocate_device(P *&ptr, int64_t const num_elems, bool const initialize)
{
  ignore(ptr);
  ignore(num_elems);
  ignore(initialize);
  throw std::runtime_error("calling allocate_device without CUDA");
}
#endif

#ifdef ASGARD_USE_CUDA
template<typename P>
void delete_device(P *&ptr)
{
  auto const success = cudaFree(ptr);
  // the device runtime may be unloaded at process shut down
  // (when static storage duration destructors are called)
  // returning a cudartUnloading error code.
  expect((success == cudaSuccess) || (success == cudaErrorCudartUnloading));
  ptr = nullptr;
}
#else
template<typename P>
void delete_device(P *&ptr)
{
  ignore(ptr);
  throw std::runtime_error("calling delete_device without CUDA");
}
#endif

#ifdef ASGARD_USE_CUDA
template<typename P>
void copy_on_device(P *const dest, P const *const source, int const num_elems)
{
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyDeviceToDevice);
  expect(success == cudaSuccess);
}
#else
template<typename P>
void copy_on_device(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_device without CUDA");
}
#endif

#ifdef ASGARD_USE_CUDA
template<typename P>
void copy_to_device(P *const dest, P const *const source, int const num_elems)
{
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyHostToDevice);
  expect(success == cudaSuccess);
}
#else
template<typename P>
void copy_to_device(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_device without CUDA");
}
#endif

#ifdef ASGARD_USE_CUDA
template<typename P>
void copy_to_host(P *const dest, P const *const source, int const num_elems)
{
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyDeviceToHost);
  expect(success == cudaSuccess);
}
#else
template<typename P>
void copy_to_host(P *const dest, P const *const source, int const num_elems)
{
  ignore(dest);
  ignore(source);
  ignore(num_elems);
  throw std::runtime_error("calling copy_to_host without CUDA");
}
#endif

template void
allocate_device(double **&ptr, int64_t const num_elems, bool const initialize);
template void
allocate_device(double *&ptr, int64_t const num_elems, bool const initialize);
template void
allocate_device(float **&ptr, int64_t const num_elems, bool const initialize);
template void
allocate_device(float *&ptr, int64_t const num_elems, bool const initialize);
template void
allocate_device(int *&ptr, int64_t const num_elems, bool const initialize);

template void delete_device(double **&ptr);
template void delete_device(double *&ptr);
template void delete_device(float **&ptr);
template void delete_device(float *&ptr);
template void delete_device(int *&ptr);

template void copy_on_device(double *const dest, double const *const source,
                             int const num_elems);
template void copy_on_device(float *const dest, float const *const source,
                             int const num_elems);
template void
copy_on_device(int *const dest, int const *const source, int const num_elems);

template void copy_to_device(double *const dest, double const *const source,
                             int const num_elems);
template void copy_to_device(float *const dest, float const *const source,
                             int const num_elems);
template void
copy_to_device(int *const dest, int const *const source, int const num_elems);

template void copy_to_host(double *const dest, double const *const source,
                           int const num_elems);
template void
copy_to_host(float *const dest, float const *const source, int const num_elems);
template void
copy_to_host(int *const dest, int const *const source, int const num_elems);

} // namespace asgard::fk
