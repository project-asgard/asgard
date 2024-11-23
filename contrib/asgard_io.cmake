###############################################################################
## IO-related support
###############################################################################

###############################################################################
## BlueBrain/HighFive (https://github.com/BlueBrain/HighFive)
#
# header-only library for a c++ interface into libhdf5
# included in the asgard repo at contrib/HighFive
###############################################################################
if (ASGARD_USE_HIGHFIVE)

  # -- first we need HDF5
  enable_language (C)

  add_library (asgard_hdf5 INTERFACE)

  # if used asked us to build HDF5
  if (ASGARD_BUILD_HDF5)
    message (STATUS "building hdf5 from source")

    set(__asgard_h5_install_prefix "${CMAKE_INSTALL_PREFIX}")
    include (ExternalProject)
    if (DEFINED CMAKE_APPLE_SILICON_PROCESSOR AND CMAKE_APPLE_SILICON_PROCESSOR STREQUAL "arm64")
      # Get HDF5 to build on Apple silicon
      ExternalProject_Add (hdf5_external
        UPDATE_COMMAND ""
        PREFIX "contrib/hdf5"
        URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/autogen.sh
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
        BUILD_COMMAND make
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND make install
      )
    else()
      ExternalProject_Add (hdf5_external
        UPDATE_COMMAND ""
        PREFIX "contrib/hdf5"
        URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
        BUILD_COMMAND make
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND make install
      )
    endif()

    # either it was already here, or we just built it here
    set (hdf5_include ${__asgard_h5_install_prefix}/include)
    set (hdf5_lib "${__asgard_h5_install_prefix}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}hdf5${CMAKE_SHARED_LIBRARY_SUFFIX}")

    target_include_directories (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_include}>)
    target_link_libraries (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_lib}>)

  else() # not building HDF5, using the find-package

    find_package (HDF5 REQUIRED)
    target_link_libraries (asgard_hdf5 INTERFACE hdf5::hdf5)

  endif ()

  # -- second, we get HighFive itself
  set (highfive_PATH ${CMAKE_SOURCE_DIR}/contrib/highfive)
  if (NOT EXISTS ${highfive_PATH}/include/highfive/H5Easy.hpp)
    execute_process (COMMAND rm -rf ${highfive_PATH})
    execute_process (COMMAND mkdir ${highfive_PATH})

    message (STATUS "downloading HighFive from github")
    execute_process (
      COMMAND git clone --depth 1 --branch v2.9.0 https://github.com/BlueBrain/HighFive .
      WORKING_DIRECTORY ${highfive_PATH}
      RESULT_VARIABLE download
      OUTPUT_QUIET
      ERROR_QUIET
      )
    if (download)
      message (FATAL_ERROR "could not download highfive")
    endif ()
  else ()
    message (STATUS "using contrib HighFive at ${highfive_PATH}")
    execute_process (
      COMMAND git fetch -t COMMAND git reset --hard v2.9.0
      WORKING_DIRECTORY ${highfive_PATH}
      RESULT_VARIABLE download
      OUTPUT_QUIET
      ERROR_QUIET
      )
  endif ()

  add_library (asgard_highfive INTERFACE)
  target_include_directories (asgard_highfive INTERFACE $<BUILD_INTERFACE:${highfive_PATH}/include>)
  target_link_libraries (asgard_highfive INTERFACE asgard_hdf5)

endif()
