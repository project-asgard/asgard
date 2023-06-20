//-----------------------------------------------------------------------------
//
// device allocation and transfer helpers
//
//-----------------------------------------------------------------------------
#pragma once

#include "build_info.hpp"
#include "lib_dispatch.hpp"
#include "tensors.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdint>

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

#ifdef ASGARD_USE_CUDA

constexpr cudaMemcpyKind getCudaMemcpyKind(resource resrc, resource oresrc)
{
  if (resrc == resource::host)
  {
    if (oresrc == resource::host)
      return cudaMemcpyHostToHost;
    else if (oresrc == resource::device)
      return cudaMemcpyDeviceToHost;
  }
  else if (resrc == resource::device)
  {
    if (oresrc == resource::host)
      return cudaMemcpyHostToDevice;
    else if (oresrc == resource::device)
      return cudaMemcpyDeviceToDevice;
  }
}

template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename = disable_for_const_view<mem>>
void copy_vector(fk::vector<P, mem, resrc> &dest,
                 fk::vector<P, omem, oresrc> const &source)
{
  expect(source.size() == dest.size());

  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(resrc, oresrc);

  auto const success =
      cudaMemcpy(dest.data(), source.data(), source.size() * sizeof(P), kind);
  expect(success == cudaSuccess);
}
#else
template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename = disable_for_const_view<mem>>
void copy_vector(fk::vector<P, mem, resrc> &dest,
                 fk::vector<P, omem, oresrc> const &source)
{
  expect(source.size() == dest.size());
  static_assert(resrc == resource::host);
  static_assert(oresrc == resource::host);
  std::copy(std::begin(source), std::end(source), std::begin(dest));
}
#endif

#ifdef ASGARD_USE_CUDA
template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc, typename = disable_for_const_view<mem>>
void copy_matrix(fk::matrix<P, mem, resrc> &dest,
                 fk::matrix<P, omem, oresrc> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());

  cudaMemcpyKind constexpr kind = getCudaMemcpyKind(resrc, oresrc);

  auto const success =
      cudaMemcpy2D(dest.data(), dest.stride() * sizeof(P), source.data(),
                   source.stride() * sizeof(P), source.nrows() * sizeof(P),
                   source.ncols(), kind);
  expect(success == cudaSuccess);
}
#else
template<typename P, mem_type mem, resource resrc, mem_type omem,
         resource oresrc>
void copy_matrix(fk::matrix<P, mem, resrc> &dest,
                 fk::matrix<P, omem, oresrc> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());
  static_assert(resrc == resource::host);
  static_assert(oresrc == resource::host);

  for (int j = 0; j < source.ncols(); ++j)
    for (int i = 0; i < source.nrows(); ++i)
      dest(i, j) = source(i, j);
}
#endif

} // namespace asgard::fk
