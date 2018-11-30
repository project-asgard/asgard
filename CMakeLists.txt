cmake_minimum_required (VERSION 3.9)
project (asgard
  VERSION 0.1.0
  LANGUAGES CXX
)

###############################################################################
## Set up the compiler and general global build options
###############################################################################
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options ("-g" "-Wall")
message (STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

# set up possible commandline input variable defaults (override with -D)
option (ASGARD_BUILD_TESTS "Build tests for asgard" ON)
option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" OFF)

###############################################################################
## External support
#
# FIXME allow user to specify a path to suitable libraries
# FIXME allow user to provide cached tarballs
###############################################################################

## BLAS/Lapack
if (NOT ASGARD_BUILD_OPENBLAS)
  find_package (LAPACK)
  if (LAPACK_FOUND)
    message (STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
    message (STATUS "LAPACK linker flags: ${LAPACK_LINKER_FLAGS}")
    set (LINALG_LIBS ${LAPACK_LIBRARIES})
  endif ()
endif ()

# if cmake couldn't find other blas/lapack, or the user asked to build openblas
if (NOT LAPACK_FOUND)
  # first check if it has already been built
  set (OpenBLAS_PATH ${CMAKE_SOURCE_DIR}/contrib/blas/openblas)
  if (EXISTS ${OpenBLAS_PATH}/lib/libopenblas.so)
    set (LINALG_LIBS ${OpenBLAS_PATH}/lib/libopenblas.so)
    message (STATUS "OpenBLAS library: ${OpenBLAS_LIBRARIES}")

  # build it if necessary
  else (NOT DEFINED OpenBLAS_LIBRARIES)
    message (STATUS "OpenBLAS not found. We'll build it from source.")

    include (ExternalProject)
    ExternalProject_Add (openblas-ext
      PREFIX contrib/blas/openblas
      URL https://github.com/xianyi/OpenBLAS/archive/v0.3.0.tar.gz
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND make PREFIX=${OpenBLAS_PATH} install
    )
    set (build_OpenBLAS TRUE)
  endif ()
endif ()

## Clara
set (Clara_PATH ${CMAKE_SOURCE_DIR}/contrib/clara)
if (NOT EXISTS ${Clara_PATH}/clara.hpp)
  message (FATAL_ERROR "clara.hpp not found. Please add at ${Clara_PATH}")
endif ()

add_library (clara INTERFACE)
target_include_directories (clara INTERFACE ${Clara_PATH})


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
  matlab_utilities
  program_options
  tensors
)

foreach (component IN LISTS components)
  add_library (${component} src/${component}.cpp)
endforeach ()

if (build_OpenBLAS)
  add_dependencies (tensors openblas-ext)
  set (LINALG_LIBS ${OpenBLAS_PATH}/lib/libopenblas.so)
endif ()

target_link_libraries (matlab_utilities PUBLIC tensors)

target_link_libraries (program_options PRIVATE clara)

target_link_libraries (tensors PRIVATE ${LINALG_LIBS})


# define the main application and its linking
add_executable (asgard src/main.cpp)

set (main_app_link_deps
  matlab_utilities
  program_options
  tensors
)

foreach (link_dep IN LISTS main_app_link_deps)
  target_link_libraries (asgard PRIVATE ${link_dep})
endforeach ()

###############################################################################
## Testing asgard
#
###############################################################################
if (ASGARD_BUILD_TESTS)
  # Prepare "Catch" library for executables that depend on it
  add_library (Catch INTERFACE)
  target_include_directories (Catch INTERFACE
    ${CMAKE_SOURCE_DIR}/testing
    ${CMAKE_SOURCE_DIR}/contrib/catch2/include
  )

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
