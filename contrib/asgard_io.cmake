###############################################################################
## IO-related support
###############################################################################

###############################################################################
## BlueBrain/HighFive (https://github.com/BlueBrain/HighFive)
#
# header-only library for a c++ interface into libhdf5
# included in the asgard repo at contrib/HighFive
###############################################################################

  add_library (asgard_hdf5 INTERFACE)

  # if used asked us to build HDF5
  if (ASGARD_BUILD_HDF5)
    message (STATUS "ASGarD will build HDF5 from source")

    enable_language (C) # HDF5 needs C

    set(__asgard_h5_install_prefix "${CMAKE_INSTALL_PREFIX}")
    include (ExternalProject)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      if (DEFINED CMAKE_APPLE_SILICON_PROCESSOR AND CMAKE_APPLE_SILICON_PROCESSOR STREQUAL "arm64")
        # Get HDF5 to build on Apple silicon
        ExternalProject_Add (hdf5_external
          UPDATE_COMMAND ""
          PREFIX "contrib/hdf5"
          URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
          DOWNLOAD_NO_PROGRESS 1
          CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/autogen.sh
          COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
          BUILD_IN_SOURCE 1
          DOWNLOAD_EXTRACT_TIMESTAMP 0
        )
      else()
        ExternalProject_Add (hdf5_external
          UPDATE_COMMAND ""
          PREFIX "contrib/hdf5"
          URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
          DOWNLOAD_NO_PROGRESS 1
          CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
          BUILD_IN_SOURCE 1
          DOWNLOAD_EXTRACT_TIMESTAMP 0
        )
      endif()
    else()
      if (DEFINED CMAKE_APPLE_SILICON_PROCESSOR AND CMAKE_APPLE_SILICON_PROCESSOR STREQUAL "arm64")
        # Get HDF5 to build on Apple silicon
        ExternalProject_Add (hdf5_external
          UPDATE_COMMAND ""
          PREFIX "contrib/hdf5"
          URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
          DOWNLOAD_NO_PROGRESS 1
          CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/autogen.sh
          COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
          BUILD_IN_SOURCE 1
        )
      else()
        ExternalProject_Add (hdf5_external
          UPDATE_COMMAND ""
          PREFIX "contrib/hdf5"
          URL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.11/src/hdf5-1.10.11.tar.bz2
          DOWNLOAD_NO_PROGRESS 1
          CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/hdf5/src/hdf5_external/configure --prefix=${__asgard_h5_install_prefix}
          BUILD_IN_SOURCE 1
        )
      endif()
    endif()

    # either it was already here, or we just built it here
    set (hdf5_include ${__asgard_h5_install_prefix}/include)
    set (hdf5_lib "${__asgard_h5_install_prefix}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}hdf5${CMAKE_SHARED_LIBRARY_SUFFIX}")

    target_include_directories (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_include}>)
    target_link_libraries (asgard_hdf5 INTERFACE $<BUILD_INTERFACE:${hdf5_lib}>)

  else() # not building HDF5, using the find-package

    find_package (HDF5 REQUIRED)
    target_link_libraries (asgard_hdf5 INTERFACE HDF5::HDF5)

  endif ()

  include(ExternalProject)

  set (__asg_highfive_url https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.10.1.tar.gz)
  set (__asg_highfive_path ${CMAKE_SOURCE_DIR}/contrib/highfive)

  if (NOT EXISTS ${__asg_highfive_path}/include/highfive/H5Easy.hpp)
    message(STATUS "Fetching content: ${__asg_highfive_url}")
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      ExternalProject_Add(
        asgard_highfive_down
        URL ${__asg_highfive_url}
        URL_HASH SHA256=60d66ba1315730494470afaf402bb40300a39eb6ef3b9d67263335a236069cce
        SOURCE_DIR ${__asg_highfive_path}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        DOWNLOAD_EXTRACT_TIMESTAMP 0
      )
    else()
      ExternalProject_Add(
        asgard_highfive_down
        URL ${__asg_highfive_url}
        URL_HASH SHA256=60d66ba1315730494470afaf402bb40300a39eb6ef3b9d67263335a236069cce
        SOURCE_DIR ${__asg_highfive_path}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
      )
    endif()

    ExternalProject_Add_StepTargets(asgard_highfive_down configure)
    add_dependencies(asgard::LINALG asgard_highfive_down-configure)
  endif()

  add_library (asgard_highfive INTERFACE)
  target_include_directories (asgard_highfive INTERFACE $<BUILD_INTERFACE:${__asg_highfive_path}/include>)
  target_link_libraries (asgard_highfive INTERFACE asgard_hdf5)
