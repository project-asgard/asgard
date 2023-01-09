cmake_minimum_required (VERSION 3.19)

project (asgard
  VERSION 0.5.0
  LANGUAGES CXX
)

#-------------------------------------------------------------------------------
#  Setup the kromult external project.
#-------------------------------------------------------------------------------
include (FetchContent)
find_package (Git)

#  Define a macro to register new projects.
function (register_project name dir url default_tag)
    message (STATUS "Registering project ${name}")

    set (BUILD_TAG_${dir} ${default_tag} CACHE STRING "Name of the tag to checkout.")
    set (BUILD_REPO_${dir} ${url} CACHE STRING "URL of the repo to clone.")

    #Check for optional patch file.
    set(PATCH_COMMAND "")
    if(${ARGC} EQUAL 5)
        find_package(Git)
        set(_apply_flags --ignore-space-change --whitespace=fix)
        set(PATCH_COMMAND "${GIT_EXECUTABLE}" reset --hard ${BUILD_TAG_${dir}} COMMAND "${GIT_EXECUTABLE}" apply ${_apply_flags} "${ARGV4}")
    endif()
    #  Set up the sub project repository.
    FetchContent_Declare(
        ${name}
        GIT_REPOSITORY ${BUILD_REPO_${dir}}
        GIT_TAG ${BUILD_TAG_${dir}}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/${dir}
        PATCH_COMMAND ${PATCH_COMMAND}
    )
    FetchContent_MakeAvailable(${name})
endfunction ()

#  Changes to the current version of kromult should proceed through a pull
#  request. By default, a specific tag should be specifed.
register_project (kronmult
                  KRONMULT
                  https://github.com/project-asgard/kronmult.git
                  580c6cbd20d21f445799ab51a20c8857f6afebfc
)

#  Catch two testing framework.
register_project (Catch2
                  CATCH2
                  https://github.com/catchorg/Catch2.git
                  v3.1.0
)

#Mark CATCH variables as advanced.
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    string(FIND "${_variableName}" "CATCH_" out)
    if("${out}" EQUAL 0)
        mark_as_advanced(${_variableName})
    endif()
endforeach()

###############################################################################
## Set up the compiler and general global build options
###############################################################################
#set (CMAKE_CXX_STANDARD 17)
#set (CMAKE_CXX_STANDARD_REQUIRED ON)

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

# set up possible commandline input variable defaults (override with -D)
include(CMakeDependentOption)
option (ASGARD_BUILD_TESTS "Build tests for asgard" ON)
option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" OFF)
option (ASGARD_IO_HIGHFIVE "Use the HighFive HDF5 header library for I/O" OFF)
option (ASGARD_BUILD_HDF5 "Download and build our own HDF5/HighFive" OFF)
option (ASGARD_HIGHFIVE_PATH "Optional location of existing HighFive library" "")
option (ASGARD_HDF5_PATH "Optional location of lib/ containing HDF5" "")
option (ASGARD_BUILD_PROFILE_DEPS "Download and build our own tools (e.g.  graphviz)" OFF)
option (ASGARD_PROFILE_GPROF "enable profiling support for using gprof" "")
option (ASGARD_PROFILE_XRAY "enable profiling support for using LLVM XRAY" "")
option (ASGARD_PROFILE_GPERF_CPU "enable cpu profiling support using gperftools" "")
option (ASGARD_PROFILE_GPERF_MEM "enable memory profiling support using gperftools" "")
option (ASGARD_PROFILE_PERF "enable profiling support for using linux perf" "")
option (ASGARD_PROFILE_VALGRIND "enable profiling support for using valgrind" "")
option (ASGARD_GRAPHVIZ_PATH "optional location of bin/ containing dot executable" "")
option (ASGARD_USE_CUDA "Optional CUDA support for asgard" OFF)
option (ASGARD_USE_OPENMP "Optional OpenMP support for asgard" OFF)
option (ASGARD_USE_MPI "Optional distributed computing support for asgard" OFF)
cmake_dependent_option (ASGARD_USE_SCALAPACK "Use optional scalapack linear algebra library" OFF "ASGARD_USE_MPI" OFF)
option (ASGARD_USE_MATLAB "Optional MATLAB plotting support for asgard" OFF)
option (ASGARD_USE_DOUBLE_PREC "enable or disable double-precision floating point" ON)
option (ASGARD_USE_PCH "Enable precompiled header files." OFF)
set (ASGARD_TESTING_RANKS "0" CACHE STRING "Override default number of ranks to use for testing")

option (ASGARD_RECOMMENDED_DEFAULTS "Enable OpenMP, set some flags, download OpenBLAS if system BLAS is missing." OFF)
option (ASGARD_BUILD_DOCS "Build the documentation." OFF)

if (${ASGARD_USE_PCH})
  set (USE_PCH ON CACHE BOOL "Turn on kronmult precompiled header support" FORCE)
endif()

if (ASGARD_RECOMMENDED_DEFAULTS)
  # add compiler flags we always want to use
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wpedantic -Wshadow")
endif()

# add scripts directory location
set(ASGARD_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts/")

#Enable HighFive IO if any of the following variables are defined
if(NOT ASGARD_IO_HIGHFIVE)
  if(ASGARD_HDF5_PATH OR ASGARD_BUILD_HDF5 OR ASGARD_HIGHFIVE_PATH)
    set(ASGARD_IO_HIGHFIVE ON CACHE BOOL "" FORCE)
  endif()
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

if(ASGARD_USE_OPENMP OR ASGARD_RECOMMENDED_DEFAULTS)
  find_package(OpenMP)

  if (OpenMP_FOUND)
    set (ASGARD_USE_OPENMP ON CACHE BOOL "OpenMP found and enabled" FORCE)
    if(ASGARD_USE_MKL)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR
          "MKL and apple clang++ require conflicting
           openmp flags - build configuration not supported")
      endif()
    endif()
  else()
    if(ASGARD_USE_OPENMP)
      message(FATAL_ERROR "CMake could not find OpenMP flags")
    endif()
  endif()

endif()

if (ASGARD_USE_MATLAB)
    find_package(Matlab REQUIRED) # Should there be a minimum MATLAB version needed?
endif ()

if (ASGARD_USE_MPI)
    find_package(MPI REQUIRED)
endif ()

if (ASGARD_USE_SCALAPACK)
#  Work around for issues that can arise when scalapack was compiler with a
#  different vendor compiler than asgard. Specifically if scalapack was compiled
#  with gfortran but asgard is being linked with clang, clang will fail to
#  include GNU runtime libraries. Enabling the Fortran language allows the run
#  time libraries to be included at link time.
    enable_language (Fortran)
    find_package (scalapack REQUIRED)
endif ()

#-------------------------------------------------------------------------------
#  Define the asgard library.
#-------------------------------------------------------------------------------
set(ASGARD_GOLD_BASE_DIR "${PROJECT_SOURCE_DIR}/testing/generated-inputs")

configure_file(
  src/build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp
)

add_library (asgard_obj)

if (build_hdf5)
    add_dependencies (asgard_obj hdf5-ext)
endif ()

target_compile_features (asgard_obj PUBLIC cxx_std_17)
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
     moment
     pde
     permutations
     program_options
     quadrature
     solver
     tensors
     time_advance
     tools
     transformations
     asgard_field
     asgard_dimension
     asgard_discretization
     asgard_pde_system
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
    target_sources (asgard_obj
                    PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${component}.cpp>
    )
    target_precompile_headers (asgard_obj
                               PUBLIC
                               $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${component}.hpp>>
    )
endforeach ()
target_sources (asgard_obj
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/kronmult_cuda.cpp>
)
target_precompile_headers (asgard_obj
                           PUBLIC
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_base.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_advection1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_continuity1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_continuity2.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_continuity3.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_continuity6.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_diffusion1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_diffusion2.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck1_4p3.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck1_4p4.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck1_4p5.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck1_pitch_C.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck1_pitch_E.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_fokkerplanck2_complete.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/pde_vlasov_lb_full_f.hpp>>
)
target_link_libraries (asgard_obj
                       PUBLIC
                       $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
                       $<$<BOOL:${scalapack_FOUND}>:scalapack>
                       $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>
                       $<$<BOOL:${LINALG_FOUND}>:LINALG::LINALG>
                       kron
                       clara
                       $<$<BOOL:${ASGARD_IO_HIGHFIVE}>:highfive>
                       $<$<BOOL:${ASGARD_IO_HIGHFIVE}>:hdf5>
                       $<$<BOOL:${ASGARD_USE_MATLAB}>:${Matlab_ENGINE_LIBRARY}>
                       $<$<BOOL:${ASGARD_USE_MATLAB}>:${Matlab_DATAARRAY_LIBRARY}>
)
target_include_directories (asgard_obj
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_compile_definitions (asgard_obj
                            PUBLIC
                            $<BUILD_INTERFACE:$<$<BOOL:${ASGARD_USE_SCALAPACK}>:ASGARD_USE_SCALAPACK>>
)

if (ASGARD_USE_CUDA)
    enable_language (CUDA)

    find_package (CUDAToolkit REQUIRED)

    set_target_properties (asgard_obj PROPERTIES CUDA_ARCHITECTURES 70)
    set_source_files_properties (${CMAKE_CURRENT_SOURCE_DIR}/src/device/kronmult_cuda.cpp
                                 PROPERTIES LANGUAGE CUDA)

#  Turn on GPU support in kronmult.
    set (USE_GPU ON CACHE BOOL "Turn on kronmult gpu support" FORCE)

    target_compile_options (asgard_obj
                            PUBLIC
                            $<$<COMPILE_LANGUAGE:CUDA>:-g -lineinfo --ptxas-options=-O3>
    )
    target_compile_features (asgard_obj PUBLIC cuda_std_14)
    target_link_options (asgard_obj
                         PUBLIC
                         $<$<COMPILE_LANGUAGE:CUDA>:-Wl,-rpath,${CMAKE_BINARY_DIR}>
    )
    target_link_libraries (asgard_obj
                           PUBLIC
                           CUDA::cudart
                           CUDA::cublas
    )
endif ()

if (ASGARD_USE_MATLAB)
    target_include_directories (asgard_obj
                                SYSTEM PUBLIC
                                $<BUILD_INTERFACE:${Matlab_INCLUDE_DIRS}>
    )
endif()

#-------------------------------------------------------------------------------
#  Define a asgard executable library.
#-------------------------------------------------------------------------------
add_executable (asgard)
target_sources (asgard
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp>
)
target_link_libraries (asgard PRIVATE asgard_obj)
if (${ASGARD_USE_PCH})
    target_precompile_headers (asgard REUSE_FROM asgard_obj)
endif ()

#-------------------------------------------------------------------------------
# Builds the documentation, must come after asgard target is defined
#-------------------------------------------------------------------------------
if (ASGARD_BUILD_DOCS)
  add_subdirectory(doxygen)
endif()

###############################################################################
## Testing asgard
#
###############################################################################
if (ASGARD_BUILD_TESTS)
  enable_testing ()

  # add sandbox executable, i.e., an executable that is part of the build system
  # contain all the appropriate link flags and dependencies, but does nothing
  # other than play with some code
  add_executable(sandbox ./testing/sandbox.cpp)
  target_link_libraries (sandbox PUBLIC asgard_obj Catch2::Catch2)

  # Define ctest tests and their executables. The _main variant of these targets
  # uses the default main function from the catch two framework. The non _main
  # variant uses a custom defined main in MPI based tests.
  add_library (tests_general testing/tests_general.cpp)
  add_library (tests_general_main testing/tests_general.cpp)
  target_link_libraries (tests_general PUBLIC asgard_obj Catch2::Catch2)
  target_link_libraries (tests_general_main PUBLIC asgard_obj Catch2::Catch2WithMain)
  target_include_directories(tests_general
                             PRIVATE ${CMAKE_BINARY_DIR}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(tests_general_main
                             PRIVATE ${CMAKE_BINARY_DIR}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if (${ASGARD_USE_PCH})
    target_precompile_headers (sandbox REUSE_FROM asgard_obj)
    target_precompile_headers (tests_general REUSE_FROM asgard_obj)
    target_precompile_headers (tests_general_main REUSE_FROM asgard_obj)
  endif ()

  # components with MPI-enabled testing
  set (mpi_test_components
    adapt
    distribution
    time_advance
    asgard_discretization
  )
  set (mpi_single_node_components # eventually there will be none
    asgard_discretization
    asgard_pde_system
  )

  if (ASGARD_USE_SCALAPACK)
    list(APPEND mpi_test_components cblacs_grid fast_math scalapack_matrix_info scalapack_vector_info)
  endif()

  foreach (component IN LISTS components)
    add_executable (${component}-tests)
    target_sources (${component}-tests PRIVATE src/${component}_tests.cpp)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_SOURCE_DIR}/testing)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_BINARY_DIR})

    if (ASGARD_USE_MPI)
      if (${component} IN_LIST mpi_test_components)
        target_link_libraries (${component}-tests PUBLIC tests_general)
        if (${ASGARD_USE_PCH})
            target_precompile_headers (${component}-tests REUSE_FROM tests_general)
        endif ()

        set(test_ranks "4")
      	if (ASGARD_USE_CUDA)
          set(test_ranks "1")
        endif ()
        if (${ASGARD_TESTING_RANKS})
         set(test_ranks ${ASGARD_TESTING_RANKS})
        endif ()
#  Avoid over subscribe errors by limiting the number of MPI processes
#  to the maximum number detected by FindMPI
        if (${test_ranks} GREATER ${MPIEXEC_MAX_NUMPROCS})
            set(test_ranks ${MPIEXEC_MAX_NUMPROCS})
        endif ()
        if (${component} IN_LIST mpi_single_node_components)
          set(test_ranks "1")
        endif()

        foreach (rank RANGE 1 ${test_ranks})
          add_test (NAME ${component}-test-mpi_${rank}
                    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${rank} ./${component}-tests
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )

#  A 10 minute timeout should be enough time to allow a normal run. If
#  the test runs longer than this, there is probably a bug or deadlock.
          set_tests_properties (${component}-test-mpi_${rank} PROPERTIES PROCESSORS ${rank}
                                                                         ENVIRONMENT OMP_NUM_THREADS=1
                                                                         TIMEOUT 600)
        endforeach ()
      else ()
        target_link_libraries (${component}-tests PUBLIC tests_general_main)
        if (${ASGARD_USE_PCH})
            target_precompile_headers (${component}-tests REUSE_FROM tests_general_main)
        endif ()
      endif ()
    else ()
      target_link_libraries (${component}-tests PUBLIC tests_general_main)
      if (${ASGARD_USE_PCH})
        target_precompile_headers (${component}-tests REUSE_FROM tests_general_main)
      endif ()
    endif ()

    add_test (NAME ${component}-test
              COMMAND ${component}-tests
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties (${component}-test PROPERTIES PROCESSORS 2 ENVIRONMENT OMP_NUM_THREADS=2)
  endforeach ()

  add_executable (kronmult_cuda-tests)
  target_sources (kronmult_cuda-tests
                  PRIVATE
                  $<BUILD_INTERFACE:src/device/kronmult_cuda_tests.cpp>)
  target_include_directories (kronmult_cuda-tests
                              PRIVATE
                              ${CMAKE_SOURCE_DIR}/testing
                              ${CMAKE_BINARY_DIR}
  )
  target_link_libraries (kronmult_cuda-tests PUBLIC tests_general_main)
  if (${ASGARD_USE_PCH})
    target_precompile_headers (kronmult_cuda-tests REUSE_FROM tests_general_main)
  endif ()
  add_test (NAME kronmult_cuda-test
            COMMAND kronmult_cuda-tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_tests_properties (kronmult_cuda-test PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
else ()
  message (WARNING "warning: tests will not be built.")
endif ()
