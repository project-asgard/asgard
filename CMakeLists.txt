cmake_minimum_required (VERSION 3.21)

project (asgard
  VERSION 0.3.0
  LANGUAGES CXX
)

#-------------------------------------------------------------------------------
#  Setup the kromult external project.
#-------------------------------------------------------------------------------
include (FetchContent)
find_package (Git)

#  Define a macro to register new projects.
function (register_project name dir url default_tag make_avail)
    set (BUILD_TAG_${dir} ${default_tag} CACHE STRING "Name of the tag to checkout.")
    set (BUILD_REPO_${dir} ${url} CACHE STRING "URL of the repo to clone.")

#  Set up the sub project repository.
    FetchContent_Declare(
        ${name}
        GIT_REPOSITORY ${BUILD_REPO_${dir}}
        GIT_TAG ${BUILD_TAG_${dir}}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/${dir}
    )

    if (${make_avail})
      FetchContent_MakeAvailable(${name})
    endif()
endfunction ()

#  Changes to the current version of kromult should proceed through a pull
#  request. By default, a specific tag should be specifed.
register_project (kronmult
                  KRONMULT
                  https://github.com/project-asgard/kronmult.git
                  f941819685bbd3026a85145dde286f593683c1f4
                  OFF
)

###############################################################################
## Set up the compiler and general global build options
###############################################################################
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
# https://blog.kitware.com/cmake-and-the-default-build-type/
set (default_build_type "Release")
if (EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set (default_build_type "Debug")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE
       STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else ()
  message (STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
endif ()

# add compiler flags we always want to use
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wpedantic")

# set up possible commandline input variable defaults (override with -D)
option (ASGARD_BUILD_TESTS "Build tests for asgard" ON)
option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" OFF)
option (ASGARD_BUILD_HDF5 "Download and build our own HDF5/HighFive" OFF)
option (ASGARD_HDF5_PATH "optional location of lib/ containing HDF5" "")
option (ASGARD_BUILD_PROFILE_DEPS "Download and build our own tools (e.g.  graphviz)" OFF)
option (ASGARD_LAPACK_PATH "optional location of include/ and lib/ containing LAPACK" "")
option (ASGARD_BLAS_PATH "optional location of include/ and lib/ containing BLAS" "")
option (ASGARD_PROFILE_GPROF "enable profiling support for using gprof" "")
option (ASGARD_PROFILE_XRAY "enable profiling support for using LLVM XRAY" "")
option (ASGARD_PROFILE_GPERF_CPU "enable cpu profiling support using gperftools" "")
option (ASGARD_PROFILE_GPERF_MEM "enable memory profiling support using gperftools" "")
option (ASGARD_PROFILE_PERF "enable profiling support for using linux perf" "")
option (ASGARD_PROFILE_VALGRIND "enable profiling support for using valgrind" "")
option (ASGARD_GRAPHVIZ_PATH "optional location of bin/ containing dot executable" "")
option (ASGARD_IO_HIGHFIVE "Use the HighFive HDF5 header library for I/O" OFF)
option (ASGARD_USE_HIP "Optional HIP support for asgard" OFF)
option (ASGARD_USE_OPENMP "Optional openMP support for asgard" ON)
option (ASGARD_USE_MPI "Optional distributed computing support for asgard" OFF)
include(CMakeDependentOption)
cmake_dependent_option(ASGARD_USE_SCALAPACK "Use optional scalapack linear algebra library" OFF "ASGARD_USE_MPI" OFF)
option (ASGARD_USE_MATLAB "Optional MATLAB plotting support for asgard" OFF)
option (ASGARD_USE_DOUBLE_PREC "enable or disable double-precision floating point" ON)
set (ASGARD_TESTING_RANKS "0" CACHE STRING "Override default number of ranks to use for testing")

# add scripts directory location
set(ASGARD_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts/")

if (NOT ASGARD_BLAS_PATH AND ASGARD_LAPACK_PATH)
  set (ASGARD_BLAS_PATH ${ASGARD_LAPACK_PATH})
endif()
if (NOT ASGARD_LAPACK_PATH AND ASGARD_BLAS_PATH)
  set (ASGARD_LAPACK_PATH ${ASGARD_BLAS_PATH})
endif()
###############################################################################
## Pull in external support as needed
###############################################################################

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
find_package (LINALG REQUIRED)

# sets OpenBLAS_PATH and LINALG_LIBS
include (${CMAKE_SOURCE_DIR}/contrib/contrib.cmake)
# sets PROFILE_DEPS and PROFILE_LIBS
include (${CMAKE_SOURCE_DIR}/profiling/profiling.cmake)
# sets HighFive_FOUND
include (${CMAKE_SOURCE_DIR}/contrib/io.cmake)

string(FIND "${BLAS_LIBRARIES}" "mkl" FOUND_MKL)
set(ASGARD_USE_MKL FALSE)
if(FOUND_MKL GREATER_EQUAL 0)
  set(ASGARD_USE_MKL TRUE)
endif()

#if (ASGARD_USE_SLATE)
#  find_package(slate)
#  set(blaspp_DIR /opt/slate/lib/blaspp/)
#  # Without at least find_package(lapackpp), we get a linking error cannot find "`-llpackpp -llblaspp"
#  find_package(blaspp)
#  set(lapackpp_DIR /opt/slate/lib/lapackpp/)
#  find_package(lapackpp)
#endif ()

if(ASGARD_USE_OPENMP)
  find_package(OpenMP REQUIRED)  
  if(ASGARD_USE_MKL)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	message(FATAL_ERROR 
	  "MKL and apple clang++ require conflicting 
	  openmp flags - build configuration not supported")
    endif()
  endif()
endif()

# convenience flags for which HIP platform has been setup
set(ASGARD_PLATFORM_NVCC 0)
set(ASGARD_PLATFORM_AMD 0)
if(ASGARD_USE_HIP)
  # search for HIP and libraries
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()

  # set HIP_CLANG_PATH for potential installs in non-standard locations (such as rocm with spack)
  if (NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
      set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to HIP clang binaries")
    else()
      set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to HIP clang binaries")
    endif()
  endif()

  # note: could probably grab this path directly using hipconfig?
  if (NOT DEFINED HIP_CLANG_INCLUDE_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_INCLUDE_PATH})
      # probably need a better way to get the compiler version.. this will cause non-existent paths for non-clang compilers
      set(HIP_CLANG_INCLUDE_PATH "${HIP_CLANG_PATH}/../lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include" CACHE PATH "Path to HIP clang include directory")
    else()
      set(HIP_CLANG_INCLUDE_PATH $ENV{HIP_CLANG_INCLUDE_PATH} CACHE PATH "Path to HIP clang include directory")
    endif()
  endif()

  if(NOT DEFINED HIPBLAS_PATH)
    if(NOT DEFINED ENV{HIPBLAS_PATH})
        set(HIPBLAS_PATH "${HIP_PATH}/../hipblas" CACHE PATH "Path to which HIPBLAS has been installed")
    else()
        set(HIPBLAS_PATH $ENV{HIPBLAS_PATH} CACHE PATH "Path to which HIPBLAS has been installed")
    endif()
  endif()

  # try to find hipconfig executable which can help detect platforms and include dirs
  find_program(ASGARD_HIPCONFIG_PATH hipconfig HINTS "${HIP_PATH}/bin")
  if(ASGARD_HIPCONFIG_PATH)
    execute_process(COMMAND ${ASGARD_HIPCONFIG_PATH} --platform OUTPUT_VARIABLE ASGARD_HIP_PLATFORM)
  elseif(DEFINED ENV{HIP_PLATFORM})
    set(ASGARD_HIP_PLATFORM "$ENV{HIP_PLATFORM}")
  else()
    message(FATAL_ERROR "Could not determine HIP platform, make sure HIP_PLATFORM is set")
  endif()

  message(STATUS "HIP platform has been detected as ${ASGARD_HIP_PLATFORM}")
  # hip >= 4.2 is now using "amd" to identify platform
  if(ASGARD_HIP_PLATFORM STREQUAL "hcc" OR ASGARD_HIP_PLATFORM STREQUAL "amd")
    set(ASGARD_PLATFORM_AMD 1)
  # hip <= 4.1 uses "nvcc" to identify nvidia platforms, >= 4.2 uses "nvidia"
  elseif(ASGARD_HIP_PLATFORM STREQUAL "nvcc" OR ASGARD_HIP_PLATFORM STREQUAL "nvidia")
    set(ASGARD_PLATFORM_NVCC 1)
  endif()

  # double check for cuda path since HIP uses it internally
  if(ASGARD_PLATFORM_NVCC)
    if (NOT DEFINED ENV{CUDA_PATH})
      find_path(ASGARD_HIP_DEFAULT_CUDA_PATH "cuda.h" PATH /usr/local/cuda/include)
      if (NOT ASGARD_HIP_DEFAULT_CUDA_PATH)
        message(FATAL_ERROR "Make sure the CUDA_PATH env is set to locate for HIP")
      endif()
    endif()
    message(STATUS "Found CUDA_PATH: $ENV{CUDA_PATH}")
  endif()

  # look for HIP cmake configs in different locations
  list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${ROCM_PATH}")
  if(ASGARD_PLATFORM_AMD)
    # note: causes issues on nvidia, but might be needed on amd platforms?
    list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake" "${ROCM_PATH}")

    # output a warning if compiling for AMD without using amd-clang
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(WARNING "Compiling HIP for AMD without using AMD clang might not work. Use -DCMAKE_CXX_COMPILER=clang++")
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${HIPBLAS_PATH}/lib/cmake")

  set(HIP_VERBOSE_BUILD ON CACHE STRING "Verbose compilation for HIP")

  set(ASGARD_HIP_FLAGS "-std=c++14;-g" CACHE STRING "HIP compiler flags for both AMD and NVIDIA")
  set(ASGARD_AMD_ARCH "906" CACHE STRING "AMD GPU architecture number")
  set(ASGARD_AMD_FLAGS "${ASGARD_HIP_FLAGS}; --amdgpu-target=gfx${ASGARD_AMD_ARCH};-O3" CACHE STRING "Flags to pass to amd-clang for HIP") # amdgpu specific options
  set(ASGARD_NVCC_ARCH "70" CACHE STRING "Nvidia architecture number") # -gencode arch=compute_xx,code=compute_xx
  set(ASGARD_NVCC_FLAGS "${ASGARD_HIP_FLAGS}; -gencode arch=compute_${ASGARD_NVCC_ARCH},code=compute_${ASGARD_NVCC_ARCH} --ptxas-options=-O3 -lineinfo" CACHE STRING "Flags to pass to NVCC") # nvcc specific options

  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} ${ASGARD_HIP_FLAGS}" CACHE STRING "")
  set(HIP_CLANG_FLAGS "${HIP_CLANG_FLAGS} ${ASGARD_AMD_FLAGS}" CACHE STRING "")
  set(HIP_NVCC_FLAGS "${HIP_NVCC_FLAGS} ${ASGARD_NVCC_FLAGS}" CACHE STRING "")

  if(ASGARD_PLATFORM_AMD)
    set(AMDGPU_TARGETS "gfx${ASGARD_AMD_ARCH}" CACHE STRING "GPU target architectures to compile for")
    set(GPU_TARGETS "gfx${ASGARD_AMD_ARCH}" CACHE STRING "GPU target architectures to compile for")

    # need a much later version for AMD since ipiv=nullptr fix not in hipblas until >4.3.1
    find_package(HIP 4.3.0 REQUIRED)
    find_package(hipblas 0.49 REQUIRED)
  else()
    find_package(HIP 4.0 REQUIRED)
    find_package(hipblas REQUIRED)
  endif()
  # Print some debug info about HIP configuration
  message(STATUS "HIP PLATFORM: ${HIP_PLATFORM}")
  message(STATUS "HIP COMPILER: ${HIP_COMPILER}")
  message(STATUS "HIP RUNTIME: ${HIP_RUNTIME}")
  message(STATUS "HIP Includes: ${HIP_INCLUDE_DIRS}")
  message(STATUS "HIP Libraries: ${HIP_LIBRARIES}")

  if(ASGARD_PLATFORM_NVCC)
    find_package(CUDA 9.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    enable_language(CUDA)
    set (CMAKE_CUDA_STANDARD 14)
    set (CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(__HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__)
  elseif(ASGARD_PLATFORM_AMD)
    #enable_language(HIP) # not yet added to latest cmake, but should be available in 3.21
    # these compile definitions should be added automatically if using amd's clang, but
    # may not necessarily be added if compiling with gcc or others
    add_compile_definitions(__HIP_PLATFORM_HCC__ __HIP_PLATFORM_AMD__)
  endif()

  if (hipBLAS_FOUND)
     message(STATUS "Found rocBLAS version ${rocBLAS_VERSION}: ${HIPBLAS_INCLUDE_DIRS}")
  endif()

  include_directories(SYSTEM ${HIP_INCLUDE_DIRS})
  # assume this include path since HIP_INCLUDE_DIRS is not being set on nvidia platform
  include_directories(SYSTEM "${HIP_PATH}/include")
  include_directories(${HIPBLAS_INCLUDE_DIRS})

  # set source file language properties
  if(ASGARD_PLATFORM_AMD)
    #set_source_files_properties( src/device/kronmult_cuda.cpp PROPERTIES LANGUAGE HIP ) # should work after cmake 3.21 release?
    set_source_files_properties( src/device/kronmult_cuda.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  elseif(ASGARD_PLATFORM_NVCC)
    set_source_files_properties( src/device/kronmult_cuda.cpp PROPERTIES LANGUAGE CUDA ) # no .cu extension
  endif()

#  Turn on GPU support in kronmult.
  set (USE_GPU ON CACHE BOOL "Turn on kronmult gpu support" FORCE)

  # pass gpu arch code to kronmult
  if(ASGARD_PLATFORM_NVCC)
    set(GPU_ARCH "${ASGARD_NVCC_ARCH}" CACHE STRING "GPU architecture code for AMD/NVIDIA" FORCE)
  else()
    set(GPU_ARCH "${ASGARD_AMD_ARCH}" CACHE STRING "GPU architecture code for AMD/NVIDIA" FORCE)
  endif()
endif()

# Fetch kronmult after configuring everything, but before adding libraries
FetchContent_MakeAvailable(kronmult)

if(ASGARD_USE_HIP)
  hip_add_library(kronmult_cuda SHARED src/device/kronmult_cuda.cpp
                  HIPCC_OPTIONS "${ASGARD_HIP_FLAGS}"
                  NVCC_OPTIONS "${ASGARD_NVCC_FLAGS}"
                  CLANG_OPTIONS "${ASGARD_AMD_FLAGS}")
  if(ASGARD_PLATFORM_NVCC)
    set_target_properties( kronmult_cuda PROPERTIES CUDA_ARCHITECTURES OFF)
    set_target_properties( kronmult_cuda PROPERTIES COMPILE_FLAGS "-arch sm_${ASGARD_NVCC_ARCH} -g -lineinfo --ptxas-options=-O3")
  endif()
  set_target_properties( kronmult_cuda PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_BINARY_DIR}")
else()
  # build component to interface with Ed's kronmult lib
  ##TODO: link to kronmult as interface library
  add_library(kronmult_cuda SHARED src/device/kronmult_cuda.cpp)
endif()

if(ASGARD_USE_MKL)
  if(ASGARD_USE_HIP AND ASGARD_PLATFORM_NVCC)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fopenmp")
  else()
    target_compile_options (kronmult_cuda PRIVATE "-fopenmp") # CMAKE doesn't handle MKL openmp link properly
    if(APPLE) # Need to link against the same openmp library as the MKL.
      foreach (_lib IN LISTS LINALG_LIBS)
        if (_lib MATCHES "libiomp5")
          target_link_libraries(kronmult_cuda PRIVATE ${_lib})
          break()
        endif()
      endforeach()
    endif()
  endif()
endif()

if(ASGARD_USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_DIRS})
endif()

if(ASGARD_USE_MATLAB)
  find_package(Matlab REQUIRED) # Should there be a minimum MATLAB version needed?
endif()

###############################################################################
## Building asgard
#
# KEEP EVERYTHING LISTED IN ALPHABETICAL ORDER
#
# to add a component:
# 1) define the component in the components list
# 2) define any link dependencies if needed
###############################################################################

# define the components and how to build and link the components together
set (components
  adapt
  basis
  batch
  boundary_conditions
  coefficients
  distribution
  elements
  fast_math
  kronmult
  lib_dispatch
  matlab_utilities
  pde
  permutations
  program_options
  quadrature
  solver
  tensors
  time_advance
  tools
  transformations
)
if (ASGARD_IO_HIGHFIVE)
  list (APPEND components io)
endif ()

if (ASGARD_USE_SCALAPACK)
  list (APPEND components scalapack_matrix_info scalapack_vector_info cblacs_grid)
endif ()

if (ASGARD_USE_MATLAB)
  list(APPEND components matlab_plot)
endif ()

foreach (component IN LISTS components)
  add_library (${component} src/${component}.cpp)
  target_include_directories (${component} PRIVATE ${CMAKE_BINARY_DIR})
  if(ASGARD_USE_MKL)
    target_compile_options (${component} PRIVATE "-fopenmp") # CMAKE doesn't handle MKL openmp link properly
  endif()
endforeach ()

if (build_hdf5)
  add_dependencies (io hdf5-ext)
endif ()

if (build_kron)
  add_dependencies (kronmult_cuda kronmult-ext)
endif ()

if (ASGARD_USE_SCALAPACK)
  target_link_libraries (tensors PRIVATE scalapack_matrix_info cblacs_grid)
  target_link_libraries (lib_dispatch PUBLIC scalapack_matrix_info cblacs_grid)
endif ()

target_link_libraries (adapt PRIVATE distribution elements program_options transformations)

target_link_libraries (basis PRIVATE matlab_utilities quadrature tensors)

target_link_libraries (batch PRIVATE coefficients elements lib_dispatch pde tensors)

target_link_libraries (boundary_conditions PRIVATE coefficients elements transformations
                       matlab_utilities fast_math )

if (ASGARD_USE_OPENMP AND NOT ASGARD_USE_MKL)
   target_link_libraries(batch PRIVATE OpenMP::OpenMP_CXX)
endif ()

target_link_libraries (coefficients
  PRIVATE pde matlab_utilities quadrature tensors transformations)

target_link_libraries(distribution PRIVATE elements lib_dispatch)
if (ASGARD_USE_MPI)
  target_link_libraries(distribution PRIVATE MPI::MPI_CXX)
endif ()

target_link_libraries (elements
  PRIVATE permutations program_options tensors)

target_link_libraries (fast_math
  PRIVATE lib_dispatch tensors)

target_link_libraries(kronmult PRIVATE batch elements kronmult_cuda tools)

if (ASGARD_USE_OPENMP AND NOT ASGARD_USE_MKL)
   target_link_libraries(kronmult PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (ASGARD_USE_HIP AND ASGARD_PLATFORM_AMD)
  target_link_libraries(kronmult_cuda PUBLIC kron hip::device)
else ()
  target_link_libraries(kronmult_cuda PUBLIC kron)
endif()

if (ASGARD_USE_OPENMP AND NOT ASGARD_USE_MKL)
   target_link_libraries(kronmult_cuda PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (ASGARD_IO_HIGHFIVE)
  target_link_libraries (io PUBLIC highfive tensors PRIVATE hdf5)
endif ()

if (ASGARD_USE_SCALAPACK)
  find_package(scalapack REQUIRED)
  # Uncomment and remove following line after issue #23 is fixed
  # https://bitbucket.org/icl/slate/issues/23
  # target_link_libraries (lib_dispatch PRIVATE slate_scalapack_api scalapack MPI::MPI_CXX)
  target_link_libraries (lib_dispatch PRIVATE scalapack MPI::MPI_CXX)
  target_compile_definitions (lib_dispatch PUBLIC ASGARD_USE_SCALAPACK)
else ()
  target_link_libraries (lib_dispatch PRIVATE LINALG::LINALG)
endif ()

if (ASGARD_USE_HIP)
  if(ASGARD_PLATFORM_AMD)
    target_link_libraries(lib_dispatch PRIVATE hip::device)
  elseif(ASGARD_PLATFORM_NVCC)
    target_link_libraries(lib_dispatch PRIVATE ${CUDA_LIBRARIES})
  endif()
  target_link_libraries(lib_dispatch PRIVATE roc::hipblas)
endif()

if (ASGARD_USE_OPENMP AND NOT ASGARD_USE_MKL)
   target_link_libraries(lib_dispatch PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (ASGARD_USE_MATLAB)
   # SYSTEM option to suppress the compiler warnings in Matlab's C++ header files
   target_include_directories(matlab_plot SYSTEM PUBLIC ${Matlab_INCLUDE_DIRS})
   target_link_libraries(matlab_plot PRIVATE ${Matlab_LIBRARIES})
endif()

target_link_libraries (matlab_utilities PUBLIC tensors)

target_link_libraries (pde PRIVATE basis matlab_utilities tensors)

target_link_libraries (permutations PRIVATE matlab_utilities tensors)

target_link_libraries (program_options PRIVATE clara distribution)

target_link_libraries (quadrature PRIVATE matlab_utilities tensors)

target_link_libraries (solver PRIVATE distribution fast_math lib_dispatch tensors)

target_link_libraries (tensors PRIVATE lib_dispatch)
if (ASGARD_USE_HIP)
  if(ASGARD_PLATFORM_AMD)
    target_link_libraries(tensors PRIVATE hip::device)
  elseif(ASGARD_PLATFORM_NVCC)
    target_link_libraries(tensors PRIVATE ${CUDA_LIBRARIES})
  endif()
  target_link_libraries (tensors PRIVATE roc::hipblas)
endif ()
if (ASGARD_USE_SCALAPACK)	
  add_compile_definitions (ASGARD_USE_SCALAPACK)
endif ()

target_link_libraries (time_advance PRIVATE adapt fast_math pde kronmult solver tensors tools boundary_conditions
                                    INTERFACE elements)

target_link_libraries (transformations
  PRIVATE basis batch distribution matlab_utilities pde program_options
  quadrature tensors)

# define the main application and its linking
add_executable (asgard src/main.cpp)

# link in components needed directly by main
set (main_app_link_deps
  adapt
  coefficients
  distribution
  elements
  matlab_utilities
  pde
  program_options
  quadrature
  tensors
  time_advance
  tools
  transformations
)
if (ASGARD_IO_HIGHFIVE)
  list (APPEND main_app_link_deps io)
endif()

if (ASGARD_USE_MPI)
  list (APPEND main_app_link_deps PRIVATE MPI::MPI_CXX)
endif()

if (ASGARD_USE_MATLAB)
  list (APPEND main_app_link_deps matlab_plot)
endif()

target_link_libraries (asgard PRIVATE ${main_app_link_deps})

# build/link anything needed for profiling if requested
if (PROFILE_DEPS)
  add_dependencies (asgard ${PROFILE_DEPS})
endif ()
target_link_libraries (asgard PRIVATE ${PROFILE_LIBS})

# Generate header file from src/build_info.hpp.in
configure_file(
  src/build_info.hpp.in
  ${CMAKE_BINARY_DIR}/build_info.hpp
)
# Include the generated build_info.hpp
target_include_directories (asgard PRIVATE ${CMAKE_BINARY_DIR})

target_include_directories (kronmult_cuda PRIVATE ${CMAKE_BINARY_DIR})

###############################################################################
## Testing asgard
#
###############################################################################
if (ASGARD_BUILD_TESTS)
  enable_testing ()

  # Define ctest tests and their executables
  add_library (tests_general testing/tests_general.cpp)
  target_link_libraries (tests_general PUBLIC Catch PRIVATE pde program_options )
  target_include_directories(tests_general PRIVATE ${CMAKE_BINARY_DIR})
  
  # components with MPI-enabled testing
  set (mpi_test_components
    adapt
    distribution
    time_advance
  )

  if (ASGARD_USE_SCALAPACK)
    list(APPEND mpi_test_components cblacs_grid fast_math scalapack_matrix_info scalapack_vector_info)
  endif()

  foreach (component IN LISTS components)
    add_executable (${component}-tests src/${component}_tests.cpp)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_SOURCE_DIR}/testing)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_BINARY_DIR})
    target_link_libraries (${component}-tests PRIVATE ${component} tests_general)
    if (ASGARD_USE_SCALAPACK)
      target_compile_definitions (${component}-tests PUBLIC ASGARD_USE_SCALAPACK)
    endif ()

    if (ASGARD_IO_HIGHFIVE AND "${component}" STREQUAL "io")
      target_link_libraries (io-tests PRIVATE highfive hdf5 tensors)
    endif ()

    if (ASGARD_USE_MPI)
      target_link_libraries (${component}-tests PRIVATE ${component} MPI::MPI_CXX)
      if (${component} IN_LIST mpi_test_components)
        set(test_ranks "4")
        if (ASGARD_USE_HIP)
          set(test_ranks "1")
        endif ()
        if (${ASGARD_TESTING_RANKS})
         set(test_ranks ${ASGARD_TESTING_RANKS})
        endif ()
        add_test (NAME ${component}-test-mpi
                  COMMAND mpirun -n ${test_ranks} ./${component}-tests
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
	set_tests_properties (${component}-test-mpi PROPERTIES PROCESSORS ${test_ranks})
      endif()
    endif ()

    add_test (NAME ${component}-test
              COMMAND ${component}-tests
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  endforeach ()

  add_executable (kronmult_cuda-tests src/device/kronmult_cuda_tests.cpp)
  target_include_directories (kronmult_cuda-tests PRIVATE ${CMAKE_SOURCE_DIR}/testing)
  target_include_directories (kronmult_cuda-tests PRIVATE ${CMAKE_BINARY_DIR})
  target_link_libraries (kronmult_cuda-tests PRIVATE coefficients kronmult_cuda tests_general)
  add_test (NAME kronmult_cuda-test
            COMMAND kronmult_cuda-tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
else ()
  message (WARNING "warning: tests will not be built.")
endif ()
