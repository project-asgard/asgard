###############################################################################
## IO-related support
###############################################################################

###############################################################################
## HDF5 (libhdf5) (https://portal.hdfgroup.org/display/support)
#
# right now, we only access this through the HighFive wrapper lib
###############################################################################
function (get_hdf5)
  add_library (hdf5 INTERFACE)

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
  endif ()

  # if cmake couldn't find other hdf5, or the user asked to build it
  if (NOT HDF5_FOUND)
    set (hdf5_contrib_path ${CMAKE_SOURCE_DIR}/contrib/hdf5)
    find_library (hdf5_search hdf5
      PATHS ${hdf5_contrib_path} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    if (NOT hdf5_search)
      message (STATUS "libhdf5 not found. We'll build it from source.")

      include (ExternalProject)
      ExternalProject_Add (hdf5-ext
        UPDATE_COMMAND ""
        PREFIX contrib/hdf5
        URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.bz2
        DOWNLOAD_NO_PROGRESS 1
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5-ext/autogen.sh
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5-ext/configure --prefix=${hdf5_contrib_path}
	BUILD_COMMAND make
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND make install
      )
      set (build_hdf5 TRUE PARENT_SCOPE)
    else ()
      message (STATUS "using contrib HDF5 found at ${hdf5_search}")
    endif ()
    # either it was already here, or we just built it here
    set (hdf5_include ${hdf5_contrib_path}/include)
    set (hdf5_lib "-L${hdf5_contrib_path}/lib -Wl,-rpath,${hdf5_contrib_path}/lib/ -lhdf5")
  endif ()

  target_include_directories (hdf5 INTERFACE ${hdf5_include})
  target_link_libraries (hdf5 INTERFACE ${hdf5_lib})
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
  if(ASGARD_HIGHFIVE_PATH AND NOT ASGARD_BUILD_HDF5)
    find_library(Highfive PATHS ${hdf5_contrib_path} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
  else()
    set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
    register_project(highfive HIGHFIVE https://github.com/BlueBrain/HighFive v2.4.1)
    message (STATUS "using contrib HighFive at ${HighFive_BINARY_DIR}")
  endif()
endif()
