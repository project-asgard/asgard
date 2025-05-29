cmake_minimum_required (VERSION 3.19)

project (asgard VERSION 0.7.1 LANGUAGES CXX)

set(ASGARD_RELEASE_INFO "(alpha) 0.7.1a1") # sync with pip-installer setup.py

include (FetchContent) # getting external dependencies

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

option (ASGARD_BUILD_TESTS "Build tests for ASGarD" ON)
option (ASGARD_USE_OPENMP "Optional OpenMP multi-threading support" OFF)
set (ASGARD_TEST_OMP_NUM_THREADS "" CACHE STRING "Sets the OMP_NUM_THREADS variable in testing (only in testing)")
option (ASGARD_USE_CUDA "Optional Nvidia/CUDA support" OFF)
option (ASGARD_USE_ROCM "Optional AMD/ROCm support" OFF)
option (ASGARD_USE_MPI "Optional distributed computing support" OFF)

if (ASGARD_USE_MPI)
  set (ASGARD_MPI_BCAST_THRESHOLD "4" CACHE STRING "Number of ranks to switch from Send/Recv to collective Bcast")
  set (ASGARD_MPI_REDUCE_THRESHOLD "4" CACHE STRING "Number of ranks to switch from Send/Recv to collective Reduce")
  set (ASGARD_TEST_MPI_OMP_NUM_THREADS "" CACHE STRING "Sets the OMP_NUM_THREADS variable in MPI testing (only in testing)")
endif()

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

option (ASGARD_USE_GITINFO "Optional read git properties and build them into the library" ON)

if (ASGARD_USE_TIMER)
  option (ASGARD_USE_FLOPCOUNTER "Counts flops as part of the timing process" OFF)
endif()

if (ASGARD_RECOMMENDED_DEFAULTS)
  # add compiler flags we always want to use
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow")
  # Improves CPU knonmult performance but adds additional compilation time.
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
endif ()

if (ASGARD_USE_CUDA AND ASGARD_USE_ROCM)
    message(FATAL_ERROR "Cannot simultaneously use ")
endif()
if (ASGARD_USE_CUDA OR ASGARD_USE_ROCM)
    set (ASGARD_MAX_NUM_GPUS "4" CACHE STRING "Maximum number of GPU devices to use simultaneously")
    if (ASGARD_MAX_NUM_GPUS LESS 1)
        message(FATAL_ERROR "ASGARD_MAX_NUM_GPUS must be at least 1, but it is set to '${ASGARD_MAX_NUM_GPUS}'")
    endif()
else()
    set (ASGARD_MAX_NUM_GPUS "0" CACHE STRING "Maximum number of GPU devices to use simultaneously" FORCE)
endif()

#Enable HighFive IO if any of the following variables are defined
if(NOT ASGARD_USE_HIGHFIVE)
  if(ASGARD_BUILD_HDF5 OR ASGARD_USE_PYTHON)
    set(ASGARD_USE_HIGHFIVE ON CACHE BOOL "" FORCE)
  endif()
endif()

###############################################################################
# set the list of the PDEs that will be used, needed to configure files
# e.g., _asgard_doc_pdes_str is used in the example file
###############################################################################

set (_asgard_doc_pdes "continuity diffusion spherical_diffusion elliptic sinwav two_stream vplb sod_shock burgers")
set (_asgard_int_pdes bound_internal interp_internal relaxation_internal varcoeff_internal misc_internal)

set (_asgard_doc_pdes_str "${_asgard_doc_pdes}")
separate_arguments(_asgard_doc_pdes)

list(APPEND _asgard_pdes ${_asgard_doc_pdes})
list(APPEND _asgard_pdes ${_asgard_int_pdes})

###############################################################################
## Pull in external support as needed
###############################################################################

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/contrib)
find_package (LINALG REQUIRED)

# enable the git properties
if (ASGARD_USE_GITINFO)
  find_package (Git)
  if (Git_FOUND)
    set(ASGARD_HAS_GITINFO ON)
  endif()
endif()

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
    foreach(_asg_pybuildfile continuity_2d.py inputs_1d.py slideshow.py)
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/example_${_asg_pybuildfile}"
                           COMMAND "${CMAKE_COMMAND}"
                           ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile} ${CMAKE_CURRENT_BINARY_DIR}/${_asg_pybuildfile}
                           DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}"
                           COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}")
        list(APPEND _pyasgard_buildstage "${CMAKE_CURRENT_BINARY_DIR}/example_${_asg_pybuildfile}")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/examples/${_asg_pybuildfile}" DESTINATION share/asgard/examples)
    endforeach()
    add_custom_target(asgard_python_testing ALL DEPENDS "${_pyasgard_buildstage}")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgardplot.sh" "${CMAKE_CURRENT_BINARY_DIR}/asgardplot.sh" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgardplot.sh" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/python/asgardrun.sh" "${CMAKE_CURRENT_BINARY_DIR}/asgardrun.sh" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/asgardrun.sh" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

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

if (ASGARD_USE_ROCM)
    if (CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR "ASGarD AMD/ROCm GPU backend requires CMake version 3.21 or newer")
    endif()
    if (ROCM_PATH)
        set(ASGARD_ROCM_PATH "${ROCM_PATH}")
    elseif (DEFINED ENV{ROCM_PATH})
        set(ASGARD_ROCM_PATH "ENV{ROCM_PATH}")
    elseif (IS_DIRECTORY /opt/rocm)
        set(ASGARD_ROCM_PATH "/opt/rocm")
    endif()
    list (APPEND CMAKE_PREFIX_PATH ${ASGARD_ROCM_PATH}/hip ${ASGARD_ROCM_PATH})
    enable_language(HIP)
    find_package(rocsolver REQUIRED)
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

if (ASGARD_USE_MPI)
  target_link_libraries(libasgard PUBLIC MPI::MPI_CXX)
endif()

target_compile_features (libasgard PUBLIC cxx_std_17)

#  components
set(asgard_components
    blas
    block_matrix
    coefficients_mats
    compute
    dimension
    discretization
    grid_1d
    indexset
    interp
    kronmult
    kronmult_common
    kronmult_cpu
    moment
    pde
    pde_functions
    program_options
    quadrature
    reconstruct
    resource_groups
    small_mats
    solver
    term_manager
    test_macros
    time_advance
    time_data
    transformations
    tools
    wavelet_basis
)

if (ASGARD_USE_HIGHFIVE)
    list (APPEND asgard_components io)
endif ()

foreach (component IN LISTS asgard_components)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${component}.cpp)
        # some components don't have .cpp files
        target_sources (libasgard
                        PRIVATE
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${component}.cpp>
        )
    endif()
    target_precompile_headers (libasgard
                               PUBLIC
                               $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${component}.hpp>>
    )
endforeach ()

target_precompile_headers (libasgard
                           PUBLIC
                           $<$<BOOL:${ASGARD_USE_PCH}>:$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/asgard_build_info.hpp>>
)
target_link_libraries (libasgard
                       PUBLIC
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
                           CUDA::cusolver
    )
endif ()

if (ASGARD_USE_ROCM)
    target_link_libraries (libasgard PUBLIC roc::rocsolver hip::host)
endif()


#-------------------------------------------------------------------------------
#  Define a asgard executables targets.
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

foreach (_asgexe IN LISTS _asgard_pdes)
    add_executable (${_asgexe} "${CMAKE_CURRENT_SOURCE_DIR}/src/pde/${_asgexe}.cpp")
    target_link_libraries (${_asgexe} PUBLIC libasgard)
    if (${_asgexe} IN_LIST _asgard_doc_pdes)
        install(TARGETS ${_asgexe} RUNTIME DESTINATION "share/asgard/pde")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/pde/${_asgexe}.cpp"
                DESTINATION "share/asgard/examples")
    endif()
endforeach()

#-------------------------------------------------------------------------------
# Builds the documentation, must come after asgard target is defined
#-------------------------------------------------------------------------------
if (ASGARD_BUILD_DOCS)
  add_subdirectory(doxygen)
endif()

###############################################################################
## Testing asgard
###############################################################################
macro(asgard_add_mpi_test)
  cmake_parse_arguments(_asgard "" "NAME;COMMAND;RANKS" "" ${ARGN} )
  add_test(${_asgard_NAME} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_asgard_RANKS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${_asgard_COMMAND} -test ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${_asgard_NAME} PROPERTIES PROCESSORS ${_asgard_RANKS} RUN_SERIAL TRUE)
  unset(_asgard_NAME)
  unset(_asgard_RANKS)
  unset(_asgard_COMMAND)

  set(ASGARD_TEST_MPI_OMP_THREADS 2)
  if (ASGARD_TEST_OMP_NUM_THREADS)
    math(EXPR _asg_mult " ${ASGARD_TEST_MPI_OMP_THREADS} * 5")
    if (_asg_mult GREATER ASGARD_TEST_OMP_NUM_THREADS)
      set(ASGARD_TEST_MPI_OMP_THREADS 1)
    endif()
  endif()

  if (ASGARD_USE_OPENMP)
    # mpi + openmp can yield too many heavy threads and cause a huge bottleneck in testing
    set_tests_properties (${_asgard_NAME} PROPERTIES
                          ENVIRONMENT OMP_NUM_THREADS=${ASGARD_TEST_MPI_OMP_THREADS})
  endif()
endmacro()

if (ASGARD_BUILD_TESTS)
  enable_testing ()

  # add sandbox executable, i.e., an executable that is part of the build system
  # contain all the appropriate link flags and dependencies, but does nothing
  # other than play with some code
  add_executable(sandbox ./testing/sandbox.cpp)
  target_link_libraries (sandbox PUBLIC libasgard)

  if (${ASGARD_USE_PCH})
    target_precompile_headers (sandbox REUSE_FROM libasgard)
  endif ()

  # first, add the unit and integration tests from the components
  foreach (component IN LISTS asgard_components)
      # some components don't have unit-tests, only end-to-end and integration
      # those are handled through PDEs and other components, can be skipped here
      if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${component}_tests.cpp)
          continue()
      endif()

      set(_asgname "asgard-${component}-tests")

      add_executable (${_asgname})
      target_sources (${_asgname} PRIVATE src/asgard_${component}_tests.cpp)

      target_include_directories (${_asgname} PRIVATE ${CMAKE_SOURCE_DIR}/testing)
      target_include_directories (${_asgname} PRIVATE ${CMAKE_BINARY_DIR})

      target_link_libraries(${_asgname} libasgard)

      if (ASGARD_BUILD_HDF5 OR ASGARD_BUILD_OPENBLAS)
          set_target_properties(${_asgname} PROPERTIES BUILD_RPATH "${CMAKE_INSTALL_PREFIX}/lib/")
      endif()

      if (${ASGARD_USE_PCH})
          target_precompile_headers (${_asgname} REUSE_FROM libasgard)
      endif ()

      add_test (NAME ${_asgname}
                COMMAND ${_asgname}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

      if (ASGARD_TEST_OMP_NUM_THREADS)
          set_tests_properties (${_asgname} PROPERTIES
              PROCESSORS ${ASGARD_TEST_OMP_NUM_THREADS}
              ENVIRONMENT OMP_NUM_THREADS=${ASGARD_TEST_OMP_NUM_THREADS})
      endif()

  endforeach()

  # the python test is on the border, uses full pde definitions
  # but focuses on the reconstruction capabilities rather than the pde
  if (ASGARD_USE_PYTHON)
    add_test(NAME asgard-pyreconstruct-test COMMAND Python::Interpreter "${CMAKE_CURRENT_BINARY_DIR}/pyasgard_test.py")
  endif()

  # if MPI is enabled, do not test all possible combinations of pdes and mpi ranks
  # some tests are naturally slower than others and since the MPI code is little
  # full MPI coverage can be achieved with a limited number of tests
  if (ASGARD_USE_MPI)
    # at least some of the test must use ranks equal or more than mpi::bcast_threshold and mpi::reduce_threshold
    # that will switch between the two algorithms, currently using 4 and 4
    set(asgard_pde_ranks_continuity 2 3 4)
    set(asgard_pde_ranks_diffusion 2)
    set(asgard_pde_ranks_spherical_diffusion 2)
    set(asgard_pde_ranks_elliptic 2 3 4)
    set(asgard_pde_ranks_sinwav 2 4)
    set(asgard_pde_ranks_bound_internal 2 3 4)
    set(asgard_pde_ranks_relaxation_internal 2)
    set(asgard_pde_ranks_two_stream 2 3 4)
  endif()

  # add the pdes and set the MPI tests
  foreach( _asg_pde ${_asgard_pdes})
      add_test (NAME asgard-pde-${_asg_pde}-test
                COMMAND ${_asg_pde} -test
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

      if (ASGARD_TEST_OMP_NUM_THREADS)
          set_tests_properties (asgard-pde-${_asg_pde}-test PROPERTIES
                                PROCESSORS ${ASGARD_TEST_OMP_NUM_THREADS}
                                ENVIRONMENT OMP_NUM_THREADS=${ASGARD_TEST_OMP_NUM_THREADS})
      endif()

      # MPI ranks are defined only if using MPI
      if (asgard_pde_ranks_${_asg_pde})
        foreach (_asgnumranks ${asgard_pde_ranks_${_asg_pde}})
          asgard_add_mpi_test(NAME asgard-pde-${_asg_pde}-mpin${_asgnumranks}  COMMAND ${_asg_pde} RANKS ${_asgnumranks})
          if (ASGARD_TEST_MPI_OMP_NUM_THREADS)
            set_tests_properties (asgard-pde-${_asg_pde}-mpin${_asgnumranks} PROPERTIES
                                  PROCESSORS ${ASGARD_TEST_MPI_OMP_NUM_THREADS}
                                  ENVIRONMENT OMP_NUM_THREADS=${ASGARD_TEST_MPI_OMP_NUM_THREADS})
          endif()
        endforeach()
      endif()

  endforeach()

# assuming testing makes us into a stand-alone project, so be more verbose
# helps keep track of flags and options and whether things we enabled correctly
  message(STATUS "")
  message(STATUS "ASGarD build options (summary):")
  foreach(_opt CMAKE_CXX_FLAGS
               ASGARD_PRECISIONS
               ASGARD_USE_OPENMP
               ASGARD_USE_MPI
               ASGARD_USE_CUDA
               ASGARD_USE_ROCM
               ASGARD_USE_PYTHON
               ASGARD_USE_HIGHFIVE
               ASGARD_USE_TIMER)
    message(STATUS "  ${_opt}=${${_opt}}")
  endforeach()
  if (ASGARD_USE_TIMER)
    message(STATUS "  ASGARD_USE_FLOPCOUNTER=${ASGARD_USE_FLOPCOUNTER}")
  endif()
  if (ASGARD_USE_CUDA)
      foreach(_opt CMAKE_CUDA_COMPILER CMAKE_CUDA_FLAGS CMAKE_CUDA_ARCHITECTURES)
          message(STATUS "  ${_opt}=${${_opt}}")
      endforeach()
  endif()
  if (ASGARD_USE_CUDA)
      foreach(_opt CMAKE_HIP_COMPILER CMAKE_HIP_FLAGS CMAKE_HIP_ARCHITECTURES)
          message(STATUS "  ${_opt}=${${_opt}}")
      endforeach()
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

# installing the headers
foreach (_asgcomp ${asgard_components})
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${_asgcomp}.hpp")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/asgard_${_asgcomp}.hpp"
            DESTINATION include)
  endif()
endforeach()
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/asgard.hpp"
              "${CMAKE_CURRENT_SOURCE_DIR}/src/asgard.h"
        DESTINATION include)
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
