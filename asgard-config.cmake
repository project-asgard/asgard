cmake_minimum_required(VERSION 3.19)

if (TARGET asgard::asgard)
    return() # exit if ASGarD has already been found and defined
endif()

@PACKAGE_INIT@

include("@__asgard_install_prefix@/lib/@CMAKE_PROJECT_NAME@/@CMAKE_PROJECT_NAME@-targets.cmake")

if ("@ASGARD_USE_MPI@" AND NOT TARGET MPI::MPI_CXX)
  if (NOT MPI_HOME AND NOT DEFINED ENV{MPI_HOME})
    if (NOT MPI_CXX_COMPILER)
      set(MPI_CXX_COMPILER "@MPI_CXX_COMPILER@")
    endif()
  endif()
  find_package(MPI REQUIRED)
endif()

if ("@ASGARD_USE_CUDA@")
  set(CMAKE_CUDA_COMPILER "@CMAKE_CUDA_COMPILER@")
  enable_language (CUDA)
  find_package (CUDAToolkit REQUIRED)
endif()

if ("@ASGARD_USE_OPENMP@")
  find_package(OpenMP REQUIRED)
endif()

if (@BLA_VENDOR@)
  set(BLA_VENDOR "@BLA_VENDOR@")
endif()
find_package (BLAS)
find_package (LAPACK)
add_library (asgard::LINALG INTERFACE IMPORTED)
target_link_libraries (asgard::LINALG
                       INTERFACE
                       $<$<BOOL:${BLAS_FOUND}>:BLAS::BLAS>
                       $<$<BOOL:${LAPACK_FOUND}>:LAPACK::LAPACK>
)

if (@ASGARD_USE_HIGHFIVE@)
  enable_language (C)
  if (@__asgard_find_hdf5@)
    find_package (HDF5 REQUIRED)
  endif()
endif()

add_executable(asgard::exe IMPORTED)
set_property(TARGET asgard::exe PROPERTY IMPORTED_LOCATION "@__asgard_install_prefix@/bin/asgard${CMAKE_EXECUTABLE_SUFFIX_CXX}")

add_library(asgard::asgard INTERFACE IMPORTED GLOBAL)
target_link_libraries(asgard::asgard INTERFACE libasgard)

set(asgard_OPENMP_FOUND "@ASGARD_USE_OPENMP@")
set(asgard_MPI_FOUND    "@ASGARD_USE_MPI@")
set(asgard_CUDA_FOUND   "@ASGARD_USE_CUDA@")
set(asgard_PYTHON_FOUND "@ASGARD_USE_PYTHON@")
set(asgard_HDF5_FOUND   "@ASGARD_USE_HIGHFIVE@")

set(_asgard_modules "")
foreach(_asgard_module OPENMP MPI CUDA)
  if (asgard_${_asgard_module}_FOUND)
    set(_asgard_modules "${_asgard_modules} ${_asgard_module}")
  endif()
endforeach()

message(STATUS "Found ASGarD: v${asgard_VERSION} with${_asgard_modules}")
unset(_asgard_modules)
unset(_asgard_module)

check_required_components(asgard)
