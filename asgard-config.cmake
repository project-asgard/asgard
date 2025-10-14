cmake_minimum_required(VERSION 3.19)

if (TARGET asgard::asgard)
    return() # exit if ASGarD has already been found and defined
endif()

@PACKAGE_INIT@

include("@__asgard_install_prefix@/@CMAKE_INSTALL_LIBDIR@/cmake/@CMAKE_PROJECT_NAME@/@CMAKE_PROJECT_NAME@-targets.cmake")

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

if ("@ASGARD_USE_ROCM@")
  list (APPEND CMAKE_PREFIX_PATH "@ASGARD_ROCM_PATH@/hip" "@ASGARD_ROCM_PATH@")
  enable_language (HIP)
  find_package(rocsolver REQUIRED)
endif()

if ("@ASGARD_USE_OPENMP@")
  find_package(OpenMP REQUIRED)
endif()

add_library (asgard::LINALG INTERFACE IMPORTED)
if ("@ASGARD_BUILD_OPENBLAS@")
  # in Debug mode, openblas will install openblas_d library
  find_library(__asg_libname NAMES openblas_d openblas
               NO_DEFAULT_PATH PATHS "@__asgard_install_prefix@/lib")

  target_link_libraries(asgard::LINALG INTERFACE "${__asg_libname}")
  target_include_directories(asgard::LINALG INTERFACE
                             "@__asgard_install_prefix@/include")
else()
  if ("@BLA_VENDOR@")
    set(BLA_VENDOR "@BLA_VENDOR@")
  endif()

  find_package (BLAS REQUIRED)
  find_package (LAPACK REQUIRED)
  target_link_libraries (asgard::LINALG INTERFACE BLAS::BLAS LAPACK::LAPACK)
endif()

if ("@ASGARD_USE_HIGHFIVE@")
  # if ASGarD build HDF5, then the C programming language is needed
  # the targets will be loaded automatically since HDF5 is part of ASGarD exports
  if ("@ASGARD_BUILD_HDF5@")
    enable_language (C)
  else()
    enable_language(C) # needed by older versions of CMake
    # if using system HDF5, then just pull it through the regular channel
    find_package (HDF5 REQUIRED)
  endif()
endif()

add_executable(asgard::exe IMPORTED)
set_property(TARGET asgard::exe PROPERTY IMPORTED_LOCATION
             "@__asgard_install_prefix@/@CMAKE_INSTALL_BINDIR@/asgard${CMAKE_EXECUTABLE_SUFFIX_CXX}")

add_library(asgard::asgard INTERFACE IMPORTED GLOBAL)
target_link_libraries(asgard::asgard INTERFACE libasgard)

set(asgard_OPENMP_FOUND "@ASGARD_USE_OPENMP@")
set(asgard_MPI_FOUND    "@ASGARD_USE_MPI@")
set(asgard_CUDA_FOUND   "@ASGARD_USE_CUDA@")
set(asgard_ROCM_FOUND   "@ASGARD_USE_ROCM@")
set(asgard_PYTHON_FOUND "@ASGARD_USE_PYTHON@")
set(asgard_HDF5_FOUND   "@ASGARD_USE_HIGHFIVE@")

set(_asgard_modules "")
foreach(_asgard_module OPENMP MPI CUDA ROCM PYTHON HDF5)
  if (asgard_${_asgard_module}_FOUND)
    set(_asgard_modules "${_asgard_modules} ${_asgard_module}")
  endif()
endforeach()
if ("${_asgard_modules}" STREQUAL "")
  set(_asgard_modules " no extra modules")
endif()

message(STATUS "Found ASGarD: v${asgard_VERSION} with${_asgard_modules}")
unset(_asgard_modules)
unset(_asgard_module)

check_required_components(asgard)
