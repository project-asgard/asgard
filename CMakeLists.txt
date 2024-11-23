cmake_minimum_required (VERSION 3.19)

project (asgard VERSION 0.7.0 LANGUAGES CXX)

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


###############################################################################
## Set up the compiler and general global build options
###############################################################################

# Set a default build type if none was specified
# https://blog.kitware.com/cmake-and-the-default-build-type/
set (default_build_type "Release")
if (EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set (default_build_type "RelWithDebInfo")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE
       STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo")
else ()
  message (STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_CURRENT_BINARY_DIR}/asgard_install")
endif()

# set the rpath to the final path with the installation
set(__asgard_install_prefix ${CMAKE_INSTALL_PREFIX})
if (SKBUILD)
    set(__asgard_install_prefix ${ASGARD_python_pip_path})
endif()

list(APPEND CMAKE_INSTALL_RPATH "${__asgard_install_prefix}/lib")
list(APPEND CMAKE_BUILD_RPATH "${__asgard_install_prefix}/lib")

# set up possible commandline input variable defaults (override with -D)
include(CMakeDependentOption)
option (BUILD_SHARED_LIBS "Build the ASGarD library with dynamic or static linking" ON)

option (ASGARD_BUILD_TESTS "Build tests for asgard" ON)
option (ASGARD_USE_OPENMP "Optional OpenMP support for asgard" OFF)
option (ASGARD_USE_CUDA "Optional CUDA support for asgard" OFF)
cmake_dependent_option (ASGARD_USE_GPU_MEM_LIMIT "Allow the ability to limit the GPU memory used by kronmult (can hurt performance)" OFF "ASGARD_USE_CUDA" OFF)
option (ASGARD_USE_MPI "Optional distributed computing support for asgard" OFF)

option (ASGARD_USE_PYTHON "Optional Python tool for post-processing, plotting and quick prototyping" OFF)
option (ASGARD_USE_HIGHFIVE "Use the HighFive HDF5 header library for I/O" OFF)

if ($ENV{ASGARD_BUILD_OPENBLAS})
    option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" ON)
else()
    option (ASGARD_BUILD_OPENBLAS "Download and build our own OpenBLAS" OFF)
endif()
option (ASGARD_BUILD_HDF5 "Download and build our own HDF5/HighFive" OFF)
option (ASGARD_HIGHFIVE_PATH "Optional location of existing HighFive library" "")
option (ASGARD_HDF5_PATH "Optional location of lib/ containing HDF5" "")

option (ASGARD_USE_PCH "Enable precompiled header files." OFF)
set (ASGARD_TESTING_RANKS "0" CACHE STRING "Override default number of ranks to use for testing")


set (ASGARD_PRECISIONS "float;double" CACHE STRING "Select floating point precision, supported values are 'float', 'double' or 'float;double'")
set_property (CACHE ASGARD_PRECISIONS PROPERTY STRINGS "float" "double" "float\;double")

option (ASGARD_USE_TIMER "Enable the builtin profiling tool" ON)
option (ASGARD_RECOMMENDED_DEFAULTS "Enable OpenMP, set some flags, download OpenBLAS if system BLAS is missing." OFF)
option (ASGARD_BUILD_DOCS "(incomplete) Build the documentation." OFF)

if (NOT ASGARD_USE_MPI AND ASGARD_USE_CUDA)
  message(FATAL_ERROR "CUDA has been temporarily disabled for the non-mpi mode")
endif()

if (ASGARD_RECOMMENDED_DEFAULTS)
  # add compiler flags we always want to use
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")
  # Improves CPU knonmult performance but adds additional compilation time.
  if (NOT ASGARD_USE_CUDA)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
  endif ()
endif ()

if (ASGARD_USE_GPU_MEM_LIMIT AND NOT ASGARD_USE_CUDA)
  message(FATAL_ERROR " ASGARD_USE_GPU_MEM_LIMIT=ON requires ASGARD_USE_CUDA=ON")
endif()

# add scripts directory location
set(ASGARD_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/scripts/")

#Enable HighFive IO if any of the following variables are defined
if(NOT ASGARD_USE_HIGHFIVE)
  if(ASGARD_BUILD_HDF5 OR ASGARD_USE_PYTHON)
    set(ASGARD_USE_HIGHFIVE ON CACHE BOOL "" FORCE)
  endif()
endif()

###############################################################################
## Pull in external support as needed
###############################################################################

if (ASGARD_BUILD_TESTS)
    # Catch2 for testing, if Catch2_ROOT is provided then look for the existing installation,
    # if Catch2_ROOT is not given, download and build together with ASGarD
    if (Catch2_ROOT OR DEFINED ENV{Catch2_ROOT})
      find_package(Catch2 REQUIRED)
    else()
      register_project (Catch2
                        CATCH2
                        https://github.com/catchorg/Catch2.git
                        v3.3.0
      )
    endif()
endif()

#Mark CATCH variables as advanced.
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    string(FIND "${_variableName}" "CATCH_" out)
    if("${out}" EQUAL 0)
        mark_as_advanced(${_variableName})
    endif()
endforeach()


list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
find_package (LINALG REQUIRED)

# sets OpenBLAS_PATH and LINALG_LIBS
include (${CMAKE_SOURCE_DIR}/contrib/contrib.cmake)
# sets HighFive_FOUND
include (${CMAKE_SOURCE_DIR}/contrib/asgard_io.cmake)

string(FIND "${BLAS_LIBRARIES}" "mkl" FOUND_MKL)
set(ASGARD_USE_MKL FALSE)
if(FOUND_MKL GREATER_EQUAL 0)
  set(ASGARD_USE_MKL TRUE)
endif()

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

if (ASGARD_USE_PYTHON)
    if (NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR "The ASGarD python module requires shared libraries, set -DBUILD_SHARED_LIBS=ON")
    endif()
    find_package(Python 3.0 REQUIRED COMPONENTS Interpreter)

    set(__asgard_pysubpath "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    if (SKBUILD)
    # scikit build compiles and install in one place, then moves the files to a different location
        if (ASGARD_osx_framework)
            set(__asgard_pysubpath "lib/python/site-packages")
        endif()
        set(__asgard_final_path "${ASGARD_python_pip_path}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}asgard${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(_asgard_python_path "${CMAKE_INSTALL_PREFIX}/${__asgard_pysubpath}")
    else()
        # regular build, all those folders are the same
        set(__asgard_final_path "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}asgard${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(_asgard_python_path "${CMAKE_INSTALL_PREFIX}/${__asgard_pysubpath}")
    endif()

    # CMake accepts YES, ON or 1 in either upper or lower case
    # some configured scripts use only ON as a valid value
    set(ASGARD_USE_PYTHON ON)

    # the python module should be available and working from both the build and install folders
    # hence, we do the configuration twice
    set(__pyasgard_libasgard_path__ "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}asgard${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_config.py"  "${CMAKE_CURRENT_BINARY_DIR}/asgard_config.py")

    # put the module and all testing files into the build tree
    foreach(_asg_pybuildfile asgard.py pyasgard_test.py sandbox.py)
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_asg_pybuildfile}"
                           COMMAND "${CMAKE_COMMAND}"
                           ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/${_asg_pybuildfile} ${CMAKE_CURRENT_BINARY_DIR}/${_asg_pybuildfile}
                           DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/${_asg_pybuildfile}"
                           COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/python/${_asg_pybuildfile}")
        list(APPEND _pyasgard_buildstage "${CMAKE_CURRENT_BINARY_DIR}/${_asg_pybuildfile}")
    endforeach()
    foreach(_asg_pybuildfile continuity_2d.py inputs_1d.py)
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/example_${_asg_pybuildfile}"
                           COMMAND "${CMAKE_COMMAND}"
                           ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile} ${CMAKE_CURRENT_BINARY_DIR}/example_${_asg_pybuildfile}
                           DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}"
                           COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}")
        list(APPEND _pyasgard_buildstage "${CMAKE_CURRENT_BINARY_DIR}/example_${_asg_pybuildfile}")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}" DESTINATION share/asgard/examples)
    endforeach()
    add_custom_target(asgard_python_testing ALL DEPENDS "${_pyasgard_buildstage}")

    # install the module without the testing files
    set(__pyasgard_libasgard_path__ "${__asgard_final_path}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_config.py"  "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_config.py")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_config.py" DESTINATION "${_asgard_python_path}")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgard.py" DESTINATION "${_asgard_python_path}")

    # MATLAB section, always installed with python but it doesn't have to run
    set(_asgardpy_exe_ "PYTHONPATH=\\\"\$PYTHONPATH:${CMAKE_CURRENT_BINARY_DIR}\\\" ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/asgard_matlab.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_file_stats.m" "${CMAKE_CURRENT_BINARY_DIR}/asgard_file_stats.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_plot1d.m" "${CMAKE_CURRENT_BINARY_DIR}/asgard_plot1d.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_plot2d.m" "${CMAKE_CURRENT_BINARY_DIR}/asgard_plot2d.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_evaluate.m" "${CMAKE_CURRENT_BINARY_DIR}/asgard_evaluate.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_cell_centers.m" "${CMAKE_CURRENT_BINARY_DIR}/asgard_cell_centers.m" @ONLY)
    set(_asgardpy_exe_ "PYTHONPATH=\\\"\$PYTHONPATH:${_asgard_python_path}\\\" ${Python_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/share/asgard/matlab/asgard_matlab.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_file_stats.m" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_file_stats.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_plot1d.m" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_plot1d.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_plot2d.m" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_plot2d.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_evaluate.m" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_evaluate.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_cell_centers.m" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_cell_centers.m" @ONLY)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/"
            DESTINATION share/asgard/matlab
            FILES_MATCHING PATTERN "*.m")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/examples/continuity_2d.m" "${CMAKE_CURRENT_BINARY_DIR}/example_continuity_2d.m" @ONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/examples/continuity_2d.m" "${CMAKE_CURRENT_BINARY_DIR}/examples/continuity_2d.m" @ONLY)
    foreach(_asg_pybuildfile continuity_2d.m)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/examples/${_asg_pybuildfile}" DESTINATION share/asgard/examples)
    endforeach()

    set(_asgard_matlab_pypath_ "${CMAKE_CURRENT_BINARY_DIR}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_matlab.py" "${CMAKE_CURRENT_BINARY_DIR}/asgard_matlab.py" @ONLY)
    set(_asgard_matlab_pypath_ "${_asgard_python_path}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgard_matlab.py" "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_matlab.py" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pyinstall/asgard_matlab.py" DESTINATION share/asgard/matlab)
endif()

if (ASGARD_BUILD_TESTS)
    foreach(_asg_testinput test_input1.txt test_input2.txt)
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}"
                          COMMAND "${CMAKE_COMMAND}"
                          ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/testing/${_asg_testinput} ${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}
                          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/testing/${_asg_testinput}"
                          COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/testing/${_asg_testinput}")
        list(APPEND _asg_testinputfiles "${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}")
    endforeach()
endif()
foreach(_asg_testinput inputs_1d_1.txt inputs_1d_2.txt)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}"
                       COMMAND "${CMAKE_COMMAND}"
                       ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_testinput} ${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}
                       DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_testinput}"
                       COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_testinput}")
    list(APPEND _asg_testinputfiles "${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput}")

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_asg_testinput} DESTINATION share/asgard/examples)
endforeach()
add_custom_target(asgard_test_inputs ALL DEPENDS "${_asg_testinputfiles}")

if (ASGARD_USE_MPI)
    find_package(MPI REQUIRED)
endif ()

if (ASGARD_USE_CUDA)
    # CUDA has to be enabled before libasgard is created
    cmake_policy(SET CMP0104 NEW)

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        if ("$ENV{CUDAARCHS}" STREQUAL "")
            # ENV{CUDAARCHS} is used to set CMAKE_CUDA_ARCHITECTURES
            # if not present and using recent CMake, switch to "native"
            set (CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "Architecture for the CUDA device.")
        endif()
    else()
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
            message(FATAL_ERROR
"If using CMake prior to 3.24 the user must specify either \
CMAKE_CUDA_ARCHITECTURES or environment variable CUDAARCHS \
and the architecture has to be set to 60 or newer, \
which is the minimum that enables double precision atomic operations. \
CMAKE_CUDA_ARCHITECTURES could be specified as empty or 'False', \
but then the appropriate CMAKE_CUDA_FLAGS must be set manually.")
        endif()
    endif()

    enable_language (CUDA)
    find_package (CUDAToolkit REQUIRED)

    # number of blocks must be set before "asgard_build_info.hpp" is configured
    set (ASGARD_NUM_GPU_THREADS "1024" CACHE STRING "Number of threads for GPU launch kernels")
    set (ASGARD_NUM_GPU_BLOCKS "300" CACHE STRING "Number of blocks for GPU launch kernels")
endif()

if (NOT ASGARD_PRECISIONS OR "${ASGARD_PRECISIONS}" STREQUAL "")
    message(FATAL_ERROR "ASGARD_PRECISIONS must be defined to either 'float', 'double' or 'float;double'")
endif()
foreach(_prec ${ASGARD_PRECISIONS})
    if ("${_prec}" STREQUAL "float")
        set(ASGARD_ENABLE_FLOAT ON)
    elseif ("${_prec}" STREQUAL "double")
        set(ASGARD_ENABLE_DOUBLE ON)
    else()
        message(FATAL_ERROR "invalid precision ${_prec}, supported types are 'float' and 'double'")
    endif()
endforeach()

set (ASGARD_NUM_QUADRATURE "10" CACHE STRING "Number of quadrature points to be used in realspace transformations.")

#-------------------------------------------------------------------------------
#  Define the asgard library.
#-------------------------------------------------------------------------------
set(ASGARD_GOLD_BASE_DIR "${PROJECT_SOURCE_DIR}/testing/generated-inputs")

configure_file(
  src/asgard_build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/asgard_build_info.hpp
)

add_library (libasgard)
set_target_properties(libasgard PROPERTIES OUTPUT_NAME "asgard"
                                           SOVERSION ${asgard_VERSION_MAJOR}
                                           VERSION   ${PROJECT_VERSION})

if (ASGARD_BUILD_HDF5)
  add_dependencies (libasgard hdf5_external)
endif()

target_compile_features (libasgard PUBLIC cxx_std_17)
set (components
     asgard_adapt
     asgard_basis
     asgard_batch
     asgard_block_matrix
     asgard_boundary_conditions
     asgard_coefficients
     asgard_distribution
     asgard_discretization
     asgard_elements
     asgard_fast_math
     asgard_indexset
     asgard_interpolation1d
     asgard_interpolation
     asgard_kronmult_matrix
     asgard_lib_dispatch
     asgard_matlab_utilities
     asgard_moment
     asgard_pde
     asgard_permutations
     asgard_program_options
     asgard_quadrature
     asgard_reconstruct
     asgard_resources
     asgard_small_mats
     asgard_sparse
     asgard_solver
     asgard_tensors
     asgard_time_advance
     asgard_tools
     asgard_transformations
)

if (ASGARD_USE_HIGHFIVE)
    list (APPEND components asgard_io)
endif ()

foreach (component IN LISTS components)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}.cpp)
        # some components don't have .cpp files
        target_sources (libasgard
                        PRIVATE
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${component}.cpp>
        )
    endif()
    target_precompile_headers (libasgard
                               PUBLIC
                               $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${component}.hpp>>
    )
endforeach ()

target_sources (libasgard
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_kronmult.cpp>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_kronmult_cpu.cpp>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_spkronmult.cpp>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_spkronmult_cpu.cpp>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_glkronmult_bcpu.cpp>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_preconditioner_gpu.cpp>
)
target_precompile_headers (libasgard
                           PUBLIC
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/asgard_build_info.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_base.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_advection1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_continuity1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_continuity2.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_continuity3.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_continuity6.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_diffusion1.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_diffusion2.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck1_4p3.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck1_4p4.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck1_4p5.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck1_pitch_C.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck1_pitch_E.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_fokkerplanck2_complete.hpp>>
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/pde/asgard_pde_vlasov_lb_full_f.hpp>>
)
target_link_libraries (libasgard
                       PUBLIC
                       $<$<BOOL:${scalapack_FOUND}>:scalapack>
                       $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>
                       $<$<BOOL:${LINALG_FOUND}>:asgard::LINALG>
                       $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
                       $<$<BOOL:${ASGARD_USE_HIGHFIVE}>:asgard_highfive>
                       $<$<BOOL:${ASGARD_USE_HIGHFIVE}>:asgard_hdf5>
)
target_include_directories (libasgard
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (ASGARD_USE_CUDA)
    set_source_files_properties (${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_kronmult.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_spkronmult.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_glkronmult_gpu.cpp
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/device/asgard_preconditioner_gpu.cpp
                                 PROPERTIES LANGUAGE CUDA)

    target_compile_features (libasgard PUBLIC cuda_std_14)
    target_link_options (libasgard
                         PUBLIC
                         $<$<COMPILE_LANGUAGE:CUDA>:-Wl,-rpath,${CMAKE_BINARY_DIR}>
    )
    target_link_libraries (libasgard
                           PUBLIC
                           CUDA::cudart
                           CUDA::cublas
                           CUDA::cusparse
    )
endif ()

#-------------------------------------------------------------------------------
#  Define a asgard executable target.
#-------------------------------------------------------------------------------
add_executable (asgard_exe)
set_target_properties(asgard_exe PROPERTIES OUTPUT_NAME "asgard")
target_sources (asgard_exe
                PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp>
)
target_link_libraries (asgard_exe PRIVATE libasgard)
if (${ASGARD_USE_PCH})
    target_precompile_headers (asgard_exe REUSE_FROM libasgard)
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
  target_link_libraries (sandbox PUBLIC libasgard Catch2::Catch2)

  # Define ctest tests and their executables. The _main variant of these targets
  # uses the default main function from the catch two framework. The non _main
  # variant uses a custom defined main in MPI based tests.
  add_library (tests_general testing/tests_general.cpp)
  add_library (tests_general_main testing/tests_general.cpp)
  target_link_libraries (tests_general PUBLIC libasgard Catch2::Catch2)
  target_link_libraries (tests_general_main PUBLIC libasgard Catch2::Catch2WithMain)
  target_include_directories(tests_general
                             PRIVATE ${CMAKE_BINARY_DIR}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(tests_general_main
                             PRIVATE ${CMAKE_BINARY_DIR}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  if (${ASGARD_USE_PCH})
    target_precompile_headers (sandbox REUSE_FROM libasgard)
    target_precompile_headers (tests_general REUSE_FROM libasgard)
    target_precompile_headers (tests_general_main REUSE_FROM libasgard)
  endif ()

  if (ASGARD_USE_MPI)
    # new kronmult benchmarking
    add_executable(asgard_kronmult_benchmark ./src/asgard_kronmult_tests.hpp ./src/asgard_kronmult_benchmark.cpp)
    target_link_libraries (asgard_kronmult_benchmark PUBLIC libasgard)
    target_include_directories (asgard_kronmult_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/testing)
    target_include_directories (asgard_kronmult_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/)
  endif()

  # components with MPI-enabled testing
  # needed for multiple ranks and mpi-init
  set (mpi_test_components
    asgard_adapt
    asgard_boundary_conditions
    asgard_distribution
    asgard_coefficients
    asgard_time_advance
  )
  # still need mpi-init but not the extra ranks
  set (mpi_single_rank_tests
    asgard_boundary_conditions
    asgard_coefficients
  )

  foreach (component IN LISTS components)
    add_executable (${component}-tests)
    target_sources (${component}-tests PRIVATE src/${component}_tests.cpp)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_SOURCE_DIR}/testing)
    target_include_directories (${component}-tests PRIVATE ${CMAKE_BINARY_DIR})

    if (ASGARD_BUILD_HDF5 OR ASGARD_BUILD_OPENBLAS)
      set_target_properties(${component}-tests PROPERTIES BUILD_RPATH "${CMAKE_INSTALL_PREFIX}/lib/")
    endif()

    if (ASGARD_USE_MPI)
      if (${component} IN_LIST mpi_test_components)
        target_link_libraries (${component}-tests PUBLIC tests_general)
        if (${ASGARD_USE_PCH})
            target_precompile_headers (${component}-tests REUSE_FROM tests_general)
        endif ()

        set(test_ranks "4")
        if (ASGARD_USE_CUDA OR ${component} IN_LIST mpi_single_rank_tests)
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

  endforeach()

  if (ASGARD_USE_PYTHON)
    add_test(NAME pyreconstruct-test COMMAND Python::Interpreter "${CMAKE_CURRENT_BINARY_DIR}/pyasgard_test.py")
  endif()

# assuming testing makes us into a stand-alone project, so be more verbose
# helps keep track of flags and options and whether things we enabled correctly
  message(STATUS "")
  message(STATUS "ASGarD build options (summary):")
  foreach(_opt CMAKE_CXX_FLAGS
                  ASGARD_PRECISIONS
                  ASGARD_USE_OPENMP
                  ASGARD_USE_MPI
                  ASGARD_USE_CUDA
                  ASGARD_USE_PYTHON
                  ASGARD_USE_HIGHFIVE
                  ASGARD_USE_TIMER)
    message(STATUS "  ${_opt}=${${_opt}}")
  endforeach()
  if (ASGARD_USE_CUDA)
    foreach(_opt CMAKE_CUDA_COMPILER CMAKE_CUDA_FLAGS ASGARD_USE_GPU_MEM_LIMIT)
      message(STATUS "  ${_opt}=${${_opt}}")
    endforeach()
  else()
    message(STATUS "  ASGARD_USE_CUDA=${ASGARD_USE_CUDA}")
  endif()
  message(STATUS "")

endif ()

#-------------------------------------------------------------------------------
# Building the examples to make sure they build but will not run here
#-------------------------------------------------------------------------------
foreach(_exfile continuity_2d inputs_1d)
  add_executable(example_${_exfile} "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_exfile}.cpp")
  target_link_libraries(example_${_exfile} libasgard)
endforeach()

#-------------------------------------------------------------------------------
# Installing the library as stand-alone
#-------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

target_include_directories(libasgard PUBLIC $<INSTALL_INTERFACE:${__asgard_install_prefix}/include>)

foreach (_asg_target asgard_highfive asgard_hdf5 libasgard asgard_exe)
  if (TARGET ${_asg_target})
    install(TARGETS ${_asg_target}
            EXPORT  "asgard-export"
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")
  endif()
endforeach()


install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.tpp")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgard_build_info.hpp"
        DESTINATION include)

configure_file(examples/CMakeLists.txt ${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt @ONLY)
configure_file(testing/TestingCMakeLists.txt ${CMAKE_CURRENT_BINARY_DIR}/testing/CMakeLists.txt @ONLY)
configure_file(testing/test_post_install.sh ${CMAKE_CURRENT_BINARY_DIR}/test_post_install.sh @ONLY)
if (ASGARD_BUILD_TESTS)
    add_custom_target(test_install COMMAND "${CMAKE_CURRENT_BINARY_DIR}/test_post_install.sh")
endif()

configure_file(asgard-env.sh ${CMAKE_CURRENT_BINARY_DIR}/asgard-env.sh @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgard-env.sh"
        DESTINATION share/asgard/)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
        DESTINATION share/asgard/examples
        FILES_MATCHING PATTERN "*.cpp")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt"
        DESTINATION share/asgard/examples)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/testing/CMakeLists.txt"
        DESTINATION share/asgard/testing)

install(EXPORT "asgard-export" DESTINATION "lib/${CMAKE_PROJECT_NAME}" FILE "${CMAKE_PROJECT_NAME}-targets.cmake")

if (TARGET asgard_highfive)
    # must install the highfive dependency
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/contrib/highfive/include/"
            DESTINATION include)
    set(__asgard_custom_hdf5 ON)
    if (TARGET hdf5::hdf5)
        set(__asgard_find_hdf5 ON)
    else()
        set(__asgard_find_hdf5 OFF)
    endif()
endif()

configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/asgard-config.cmake"
                              "${CMAKE_CURRENT_BINARY_DIR}/asgard-config.cmake"
                              INSTALL_DESTINATION "lib/asgard/")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/asgard-config-version.cmake"
                                 COMPATIBILITY AnyNewerVersion)
# INSTALL_DESTINATION above seems to refer to something else, asgard-config.cmake must be installed explicitly
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgard-config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/asgard-config-version.cmake"
        DESTINATION "lib/asgard/")
