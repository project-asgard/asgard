

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
# Generate header file from src/build_info.hpp.in
configure_file(
  src/build_info.hpp.in
  ${CMAKE_BINARY_DIR}/build_info.hpp
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
    endif ()
  endif ()

  # search for blas/lapack packages providing cmake/pkgconfig
  if (NOT LINALG_LIBS)
    find_package (LAPACK QUIET)
    find_package (BLAS QUIET)
    if (LAPACK_FOUND)
    #if (LAPACK_FOUND AND BLAS_FOUND)
      set (LINALG_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    endif ()
  endif ()

  # search for blas/lapack libraries
  if (NOT LINALG_LIBS)
    find_library (LAPACK_LIB lapack openblas)
    find_library (BLAS_LIB blas openblas)
    if (LAPACK_LIB AND BLAS_LIB)
      set (LINALG_LIBS ${LAPACK_LIB} ${BLAS_LIB})
    endif()
  endif ()

  message (STATUS "LINALG libraries found: ${LINALG_LIBS}")
endif ()

# if cmake couldn't find other blas/lapack, or the user asked to build openblas
if (NOT LINALG_LIBS)
  # first check if it has already been built
  set (OpenBLAS_PATH ${CMAKE_SOURCE_DIR}/contrib/blas/openblas)
  find_library (LINALG_LIBS openblas PATHS ${OpenBLAS_PATH}/lib)
  if (LINALG_LIBS)
    message (STATUS "OpenBLAS library: ${LINALG_LIBS}")

  # build it if necessary
  else (NOT DEFINED LINALG_LIBS)
    message (STATUS "OpenBLAS not found. We'll build it from source.")

    include (ExternalProject)
    ExternalProject_Add (openblas-ext
      UPDATE_COMMAND ""
      PREFIX contrib/blas/openblas
      URL https://github.com/xianyi/OpenBLAS/archive/v0.3.4.tar.gz
      DOWNLOAD_NO_PROGRESS 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND make PREFIX=${OpenBLAS_PATH} install
    )
    set (build_OpenBLAS TRUE)
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
  )
endif ()
