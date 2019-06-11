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

if (NOT ASGARD_BLAS_PATH AND ASGARD_LAPACK_PATH)
  set (ASGARD_BLAS_PATH ${ASGARD_LAPACK_PATH})
endif()
if (NOT ASGARD_LAPACK_PATH AND ASGARD_BLAS_PATH)
  set (ASGARD_LAPACK_PATH ${ASGARD_BLAS_PATH})
endif()


###############################################################################
## Pull in external support as needed
###############################################################################
# sets OpenBLAS_PATH and LINALG_LIBS
include (${CMAKE_SOURCE_DIR}/contrib/contrib.cmake)
# sets PROFILE_DEPS and PROFILE_LIBS
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
  basis
  batch
  fast_math
  lib_dispatch
  coefficients
  connectivity
  element_table
  matlab_utilities
  mem_usage
  permutations
  pde
  program_options
  quadrature
  tensors
  time_advance
  transformations
)

foreach (component IN LISTS components)
  add_library (${component} src/${component}.cpp)
endforeach ()

if (build_OpenBLAS)
  add_dependencies (tensors openblas-ext)
  set (LINALG_LIBS "-L${OpenBLAS_PATH}/lib -Wl,-rpath,${OpenBLAS_PATH}/lib/ -lopenblas")
endif ()

target_link_libraries (basis PRIVATE matlab_utilities quadrature tensors)

target_link_libraries (batch PRIVATE lib_dispatch coefficients connectivity element_table pde tensors)

target_link_libraries (lib_dispatch PRIVATE ${LINALG_LIBS})

target_link_libraries (coefficients
  PRIVATE pde matlab_utilities quadrature tensors transformations)

target_link_libraries (connectivity
  PRIVATE element_table matlab_utilities permutations tensors)

target_link_libraries (element_table
  PRIVATE permutations program_options tensors)

target_link_libraries (matlab_utilities PUBLIC tensors)

target_link_libraries (pde PRIVATE basis matlab_utilities tensors)

target_link_libraries (permutations PRIVATE matlab_utilities tensors)

target_link_libraries (program_options PRIVATE clara)

target_link_libraries (quadrature PRIVATE matlab_utilities tensors)

target_link_libraries (tensors PRIVATE lib_dispatch)

target_link_libraries (time_advance PRIVATE batch fast_math pde tensors)

target_link_libraries (transformations
  PRIVATE connectivity matlab_utilities pde program_options
  quadrature tensors)

# define the main application and its linking
add_executable (asgard src/main.cpp)

# link in components needed directly by main
set (main_app_link_deps
  batch
  coefficients
  connectivity
  element_table
  matlab_utilities
  mem_usage
  pde
  program_options
  quadrature
  tensors
  time_advance
  transformations
)
foreach (link_dep IN LISTS main_app_link_deps)
  target_link_libraries (asgard PRIVATE ${link_dep})
endforeach ()

# build/link anything needed for profiling if requested
if (PROFILE_DEPS)
  add_dependencies (asgard ${PROFILE_DEPS})
endif ()
target_link_libraries (asgard PRIVATE ${PROFILE_LIBS})

# Include the generated build_info.hpp
target_include_directories (asgard PRIVATE ${CMAKE_BINARY_DIR})

###############################################################################
## Testing asgard
#
###############################################################################
if (ASGARD_BUILD_TESTS)
  enable_testing ()

  # Define ctest tests and their executables
  add_library (tests_general testing/tests_general.cpp)
  target_link_libraries (tests_general PUBLIC Catch PRIVATE pde program_options)
  foreach (component IN LISTS components)
    add_executable (${component}-tests src/${component}_tests.cpp)
    target_sources (${component}-tests PRIVATE src/${component}_tests.cpp)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_SOURCE_DIR}/testing)
    get_target_property(linkdeps ${component} LINK_LIBRARIES)
    target_link_libraries (${component}-tests PRIVATE ${component} tests_general)
    add_test (NAME ${component}-test
              COMMAND ${component}-tests
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  endforeach ()

else ()
  message (WARNING "warning: tests will not be built.")
endif ()
