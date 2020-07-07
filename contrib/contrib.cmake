

###############################################################################
## External support
###############################################################################


###############################################################################
## Add git branch and abbreviated git commit hash to git.hpp header file
###############################################################################
# Get the latest abbreviated commit hash of the working branch
# and force a cmake reconfigure
include(contrib/GetGitRevisionDescription.cmake)
get_git_head_revision(GIT_REFSPEC GIT_COMMIT_HASH)
# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Get the summary of the last commit
execute_process(
  COMMAND git log -n 1 --format="\\n  Date: %ad\\n      %s"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_SUMMARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Replace newlines with '\n\t' literal
string(REGEX REPLACE "(\r?\n)+"
       "\\\\n" GIT_COMMIT_SUMMARY
       ${GIT_COMMIT_SUMMARY}
)
# Remove double quotes
string(REGEX REPLACE "\""
       "" GIT_COMMIT_SUMMARY
       ${GIT_COMMIT_SUMMARY}
)
# Get the current date and time of build
execute_process(
  COMMAND date "+%A, %B %d %Y at %l:%M %P"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE BUILD_TIME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

###############################################################################
## Blas/Lapack
#
# FIXME allow user to provide cached tarballs
#
# For blas/lapack, the order of preference is:
#  1. build openblas if requested
#  2. look under the user-supplied path(s), if provided
#  3. search for blas/lapack libs providing cmake/pkgconfig (e.g. openblas)
#  4. search the system for libblas.so and liblapack.so
#  5. if no blas and/or lapack found, then build openblas
#
#  if cmake needs to "build openblas", and it remains from a previous build,
#  cmake will skip the build and reuse it
###############################################################################

## BLAS/Lapack
if (NOT ASGARD_BUILD_OPENBLAS)
  # search for blas/lapack packages under user-supplied path(s)
  if (ASGARD_LAPACK_PATH)
    find_library (LAPACK_LIB lapack openblas PATHS ${ASGARD_LAPACK_PATH}
      PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    find_library (BLAS_LIB blas openblas PATHS ${ASGARD_BLAS_PATH}
      PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    if (LAPACK_LIB AND BLAS_LIB)
      set (LINALG_LIBS ${LAPACK_LIB} ${BLAS_LIB})
      set (LINALG_LIBS_FOUND TRUE)
    endif ()
  endif ()

  # search for blas/lapack packages providing cmake/pkgconfig
  if (NOT LINALG_LIBS_FOUND)
    find_package (LAPACK QUIET)
    find_package (BLAS QUIET)
    #if (LAPACK_FOUND AND BLAS_FOUND)
    if (LAPACK_FOUND)
      # CMake 3.16 fixed LAPACK and BLAS detection on Cray systems, and
      # intentionally sets LAPACK_LIBRARIES and BLAS_LIBRARIES to empty
      # strings, trusting that the compiler wrapper will link the correct
      # library which supplies BLAS and LAPACK functions. If we try to append
      # LAPACK_LIBRARIES and BLAS_LIBRARIES to LINALG_LIBS whenever
      # LAPACK_FOUND is true, then LINALG_LIBS becomes itself an empty string
      # on Cray systems, and as a result will build OpenBLAS (which we don't
      # want).
      # So indicate that it was found, but the libraries remain empty for CMake
      set (LINALG_LIBS_FOUND TRUE)
      set (LINALG_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    endif ()
  endif ()

  # search for blas/lapack libraries
  if (NOT LINALG_LIBS_FOUND)
    find_library (LAPACK_LIB lapack openblas)
    find_library (BLAS_LIB blas openblas)
    if (LAPACK_LIB AND BLAS_LIB)
      set (LINALG_LIBS ${LAPACK_LIB} ${BLAS_LIB})
      set (LINALG_LIBS_FOUND TRUE)
    endif()
  endif ()

  if (LINALG_LIBS)
    message (STATUS "LINALG libraries found: ${LINALG_LIBS}")
  elseif (LINALG_LIBS_FOUND AND NOT LINALG_LIBS)
    message (STATUS "LINALG libraries found, relying on compiler wrappers")
  endif ()
endif ()

# if cmake couldn't find other blas/lapack, or the user asked to build openblas
if (NOT LINALG_LIBS_FOUND)
  # first check if it has already been built
  set (OpenBLAS_PATH ${CMAKE_SOURCE_DIR}/contrib/blas/openblas)
  find_library (LINALG_LIBS openblas PATHS ${OpenBLAS_PATH}/lib)
  if (LINALG_LIBS)
    message (STATUS "OpenBLAS library: ${LINALG_LIBS}")

  # build it if necessary
  else (NOT DEFINED LINALG_LIBS)
    message (STATUS "OpenBLAS not found. We'll build it from source.")

    # OpenBLAS will build without Fortran, but it will silently neglect to build
    # vital functions, so ensure we have Fortran to avoid this
    include( CheckLanguage )
    message( STATUS "Checking for Fortran compiler..." )
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
      enable_language(Fortran)
      message( STATUS "Fortran compiler found" )
    else()
      message( FATAL_ERROR "Fortran compiler missing - required to build OpenBLAS" )
    endif()

    include (ExternalProject)
    ExternalProject_Add (openblas-ext
      UPDATE_COMMAND ""
      PREFIX contrib/blas/openblas
      URL https://github.com/xianyi/OpenBLAS/archive/v0.3.4.tar.gz
      DOWNLOAD_NO_PROGRESS 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make USE_OPENMP=1
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND make PREFIX=${OpenBLAS_PATH} install
    )
    set (build_OpenBLAS TRUE)
    set (LINALG_LIBS "-L${OpenBLAS_PATH}/lib -Wl,-rpath,${OpenBLAS_PATH}/lib/ -lopenblas")
  endif ()
endif ()

###############################################################################
## Clara
###############################################################################
set (Clara_PATH ${CMAKE_SOURCE_DIR}/contrib/clara)
if (NOT EXISTS ${Clara_PATH}/clara.hpp)
  message (FATAL_ERROR "clara.hpp not found. Please add at ${Clara_PATH}")
endif ()

add_library (clara INTERFACE)
target_include_directories (clara INTERFACE ${Clara_PATH})


###############################################################################
## Catch2
###############################################################################
if (ASGARD_BUILD_TESTS)
  # Prepare "Catch" library for executables that depend on it
  add_library (Catch INTERFACE)
  target_include_directories (Catch INTERFACE
    ${CMAKE_SOURCE_DIR}/testing
    ${CMAKE_SOURCE_DIR}/contrib/catch2/include
    ${CMAKE_SOURCE_DIR}
  )
endif ()



###############################################################################
## E.D.'s kronmult library
#
# link to Ed D'Azevedo's kronmult library, or download/build if not present
#
###############################################################################

set(KRON_PATH "${CMAKE_SOURCE_DIR}/contrib/kronmult/src/kronmult-ext")
find_library(KRON_LIB kron PATHS ${KRON_PATH})

if(NOT KRON_LIB)
    message("-- kronmult library not found - dl and build from src")
    
    set(KRON_INC_PATH ${KRON_PATH}/make.inc.cpu)
    set(KRON_ARGS "")
    if(ASGARD_USE_CUDA)
       message("-- build with CUDA support")
       set(KRON_INC_PATH ${KRON_PATH}/make.inc.gpu)
       set(KRON_ARGS -DUSE_GPU=1)
    else()
       message("-- build without CUDA support")
    endif()

    
    include (ExternalProject)
    ExternalProject_Add (kronmult-ext
      UPDATE_COMMAND ""
      PREFIX ${CMAKE_SOURCE_DIR}/contrib/kronmult
      URL https://github.com/project-asgard/kronmult/archive/1.2.1.tar.gz
      DOWNLOAD_NO_PROGRESS 1
      CMAKE_ARGS ${KRON_ARGS}
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND ""
    )

    set (build_kron TRUE)
    set (KRON_LIB "-L${KRON_PATH} -lkron")

endif()

