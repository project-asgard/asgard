###############################################################################
## IO-related support
###############################################################################

###############################################################################
## HDF5 (libhdf5) (https://portal.hdfgroup.org/display/support)
#
# right now, we only access this through the HighFive wrapper lib
###############################################################################
function (get_hdf5)
  add_library (asgard_hdf5 INTERFACE)

  if (NOT ASGARD_BUILD_HDF5)
    # search for hdf5 under user-supplied path(s)
    if (ASGARD_HDF5_PATH)
      find_library (hdf5_search hdf5
        PATHS ${ASGARD_HDF5_PATH} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
      set (hdf5_include ${ASGARD_HDF5_PATH}/include)
      set (hdf5_lib "-L${ASGARD_HDF5_PATH}/lib -Wl,-rpath,${ASGARD_HDF5_PATH}/lib/ -lhdf5")
      message (STATUS "using external hdf5 found at ${ASGARD_HDF5_PATH}")
      set (HDF5_FOUND TRUE)
    endif ()

    # search for hdf5 in some typical locations
    if (NOT HDF5_FOUND)
      find_package (HDF5 QUIET)
      set (hdf5_include ${HDF5_INCLUDE_DIRS})
      set (hdf5_lib ${HDF5_LIBRARIES})
      message (STATUS "using external hdf5 found at ${HDF5_LIBRARIES}")
    endif ()

    # never build HDF5 unless ASGARD_BUILD_HDF5 is explicitly ON
    if (NOT HDF5_FOUND)
      message(FATAL_ERROR "could not find HDF5, plese provide -DASGARD_HDF5_PATH")
    endif()

    target_include_directories (asgard_hdf5 INTERFACE ${hdf5_include})
    target_link_libraries (asgard_hdf5 INTERFACE ${hdf5_lib})
  endif ()

  # if used asked us to build HDF5
  if (ASGARD_BUILD_HDF5)
    message (STATUS "building hdf5 from source")

    include (ExternalProject)
    if (DEFINED CMAKE_APPLE_SILICON_PROCESSOR AND CMAKE_APPLE_SILICON_PROCESSOR STREQUAL "arm64")
      # Get HDF5 to build on Apple silicon
      ExternalProject_Add (hdf5_external
        UPDATE_COMMAND ""
        PREFIX "contrib/hdf5"
        URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/autogen.sh
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${CMAKE_INSTALL_PREFIX}
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
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${CMAKE_INSTALL_PREFIX}
        BUILD_COMMAND make
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND make install
      )
    endif()

    # either it was already here, or we just built it here
    set (hdf5_include ${CMAKE_INSTALL_PREFIX}/include)
    set (hdf5_lib "-L${CMAKE_INSTALL_PREFIX}/lib -Wl,-rpath,${hdf5_contrib_path}/lib/ -lhdf5")

    target_include_directories (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_include}>)
    target_link_libraries (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_lib}>)
  endif ()

endfunction()

###############################################################################
## BlueBrain/HighFive (https://github.com/BlueBrain/HighFive)
#
# header-only library for a c++ interface into libhdf5
# included in the asgard repo at contrib/HighFive
###############################################################################
if (ASGARD_IO_HIGHFIVE)

  # first we need HDF5
  enable_language (C)
  get_hdf5()

  # now HighFive itself

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
