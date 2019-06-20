###############################################################################
## IO-related support
###############################################################################

###############################################################################
## HDF5 (libhdf5) (https://portal.hdfgroup.org/display/support)
#
# right now, we only access this through the HighFive wrapper lib
###############################################################################
function (get_hdf5)
  if (NOT ASGARD_BUILD_HDF5)
    # search for hdf5 under user-supplied path(s)
    if (ASGARD_HDF5_PATH)
      find_library(HDF5_LIB_PATH libhdf5 PATHS ${ASGARD_HDF5_PATH}
        PATH_SUFFIXES lib NO_DEFAULT_PATH)
      if (HDF5_LIB_PATH)
        set (hdf5_PATH ${HDF5_LIB_PATH})
      endif ()
    endif ()

    # search for hdf5 in some typical locations
    if (NOT hdf5_PATH)
      find_library (HDF5_LIB_PATH libhdf5 PATHS /usr/ /usr/local/
        PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
      if (HDF5_LIB_PATH)
        set (hdf5_PATH ${HDF5_LIB_PATH})
      endif ()
    endif ()

    if (hdf5_PATH)
      message (STATUS "using external hdf5 found at ${hdf5_PATH}")
    endif ()
  endif ()

  # if cmake couldn't find other hdf5, or the user asked to build it
  if (NOT hdf5_PATH)
    set (hdf5_PATH ${CMAKE_SOURCE_DIR}/contrib/hdf5)
      if (NOT EXISTS ${hdf5_PATH}/lib/libhdf5.so)
        message (STATUS "libhdf5 not found. We'll build it from source.")

        include (ExternalProject)
        ExternalProject_Add (hdf5-ext
          UPDATE_COMMAND ""
          PREFIX contrib/hdf5
          URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.bz2
          DOWNLOAD_NO_PROGRESS 1
          CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5-ext/configure --prefix=${hdf5_PATH}
          BUILD_COMMAND make
          BUILD_IN_SOURCE 1
          INSTALL_COMMAND make install
        )
      else ()
        message (STATUS "using contrib HDF5 found at ${hdf5_PATH}")
      endif ()
  endif ()
endfunction()

###############################################################################
## BlueBrain/HighFive (https://github.com/BlueBrain/HighFive)
#
# header-only library for a c++ interface into libhdf5
# included in the asgard repo at contrib/HighFive
###############################################################################
if (ASGARD_IO_HIGHFIVE)

  get_hdf5()

  find_package (HighFive REQUIRED PATHS ${CMAKE_SOURCE_DIR}/contrib/highfive)
endif()
