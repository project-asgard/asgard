cmake_minimum_required (VERSION 3.13)
project (asgard
  VERSION 0.1.0
  LANGUAGES CXX
)

###############################################################################
## Set up the compiler and general global build options
###############################################################################
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options ("-g" "-Wall" "-Wextra" "-Wpedantic")
message (STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# set up possible commandline input variable defaults (override with -D)
option (ASGARD_BUILD_TESTS "Build tests for asgard" ON)
option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" OFF)
option (ASGARD_LAPACK_PATH "location of include/ and lib/ containing LAPACK" "")
option (ASGARD_BLAS_PATH "location of include/ and lib/ containing BLAS" "")
option (ASGARD_PROFILE_GPROF "enable profiling support for using gprof" "")
option (ASGARD_PROFILE_XRAY "enable profiling support for using LLVM XRAY" "")
option (ASGARD_PROFILE_GPERF "enable profiling support for using gperftools" "")
option (ASGARD_PROFILE_VALGRIND "enable profiling support for using valgrind" "")

if (NOT ASGARD_BLAS_PATH AND ASGARD_LAPACK_PATH)
  set (ASGARD_BLAS_PATH ${ASGARD_LAPACK_PATH})
endif()
if (NOT ASGARD_LAPACK_PATH AND ASGARD_BLAS_PATH)
  set (ASGARD_LAPACK_PATH ${ASGARD_BLAS_PATH})
endif()

###############################################################################
## Pull in external support as needed
###############################################################################
include (${CMAKE_SOURCE_DIR}/contrib/contrib.cmake)
include (${CMAKE_SOURCE_DIR}/profiling/profiling.cmake)

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
  coefficients
  connectivity
  element_table
  matlab_utilities
  permutations
  pde
  program_options
  quadrature
  tensors
  transformations
)

foreach (component IN LISTS components)
  add_library (${component} src/${component}.cpp)
endforeach ()

if (build_OpenBLAS)
  add_dependencies (tensors openblas-ext)
  set (LINALG_LIBS ${OpenBLAS_PATH}/lib/libopenblas.so)
endif ()

target_link_libraries (coefficients
  PRIVATE pde matlab_utilities quadrature tensors transformations)

target_link_libraries (connectivity
  PRIVATE element_table matlab_utilities permutations tensors)

target_link_libraries (element_table
  PRIVATE permutations program_options tensors)

target_link_libraries (matlab_utilities PUBLIC tensors)

target_link_libraries (pde PRIVATE matlab_utilities tensors)

target_link_libraries (permutations PRIVATE matlab_utilities tensors)

target_link_libraries (program_options PRIVATE clara)

target_link_libraries (quadrature PRIVATE matlab_utilities tensors)

target_link_libraries (tensors PRIVATE ${LINALG_LIBS})

target_link_libraries (transformations
  PRIVATE connectivity matlab_utilities pde program_options
  quadrature tensors)

# define the main application and its linking
add_executable (asgard src/main.cpp)

set (main_app_link_deps
  coefficients
  connectivity
  element_table
  matlab_utilities
  pde
  program_options
  quadrature
  tensors
  transformations
)

foreach (link_dep IN LISTS main_app_link_deps)
  target_link_libraries (asgard PRIVATE ${link_dep})
endforeach ()

###############################################################################
## Testing asgard
#
###############################################################################
if (ASGARD_BUILD_TESTS)
  # Make tests executable
  add_executable (tests testing/tests_general.cpp)
  foreach (component IN LISTS components)
    target_sources (tests PRIVATE src/${component}_tests.cpp)
  endforeach ()

  target_link_libraries (tests PRIVATE Catch)
  foreach (link_dep IN LISTS main_app_link_deps)
    target_link_libraries (tests PRIVATE ${link_dep})
  endforeach ()

  enable_testing ()
  add_test (all-tests tests)
else ()
  message (WARNING "warning: tests will not be built.")
endif ()
