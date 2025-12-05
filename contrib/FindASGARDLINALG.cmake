#[==[
Find package that wraps functionality to find the BLAS/LAPACK libraries.

if (ASGARD_BUILD_OPENBLAS) then it will download and build OpenBLAS

Otherwise, it will use the native CMake find_package.

If find_package() finds MKL then this defined variable set(ASGARD_USING_MKL ON)

The module defines the target

    asgard::LINALG

#]==]

include (FindPackageHandleStandardArgs)

#-------------------------------------------------------------------------------
#  Setup and build OpenBLAS if ASGARD_BUILD_OPENBLAS is ON
#  otherwise use the CMake native find_package(BLAS)
#-------------------------------------------------------------------------------
if (ASGARD_BUILD_OPENBLAS)

    set (__asg_openblas_url https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz)

    message(STATUS "Fetching content: ${__asg_openblas_url}")
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      FetchContent_Declare(openblas
                           URL ${__asg_openblas_url}
                           SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS
                           DOWNLOAD_EXTRACT_TIMESTAMP 0
                           )
    else()
      FetchContent_Declare(openblas
                           URL ${__asg_openblas_url}
                           SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS
                           )
    endif()
    FetchContent_MakeAvailable(openblas)

#  Fetch content does not run the install phase so the headers for openblas are
#  not geting copied to the openblas-build directory. We will do this manually
#  instead.
    set (asgard_openblas_headers
         cblas.h
         common.h
         common_zarch.h
         common_alpha.h
         common_arm.h
         common_arm64.h
         common_c.h
         common_d.h
         common_ia64.h
         common_interface.h
         common_lapack.h
         common_level1.h
         common_level2.h
         common_level3.h
         common_linux.h
         common_macro.h
         common_mips.h
         common_mips64.h
         common_param.h
         common_power.h
         common_q.h
         common_reference.h
         common_riscv64.h
         common_s.h
         common_sb.h
         common_sparc.h
         common_stackalloc.h
         common_thread.h
         common_x.h
         common_x86_64.h
         common_x86.h
         common_z.h
         param.h
    )

    foreach (__asg_header IN LISTS asgard_openblas_headers)
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS/${__asg_header}
                        ${FETCHCONTENT_BASE_DIR}/openblas-build/${__asg_header}
                        COPYONLY)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS/${__asg_header}
                DESTINATION include/)
    endforeach ()

    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/_deps/openblas-build/lib/"
            DESTINATION lib)

    set (BLAS_FOUND 1)
    set (LAPACK_FOUND 1)

    add_library (asgard::LINALG INTERFACE IMPORTED)
    target_link_libraries (asgard::LINALG INTERFACE openblas)

#  Manually set the openblas include directory since openblas only sets the
#  include directory for the install.
    target_include_directories (asgard::LINALG
                                INTERFACE
                                ${FETCHCONTENT_BASE_DIR}/openblas-build
    )

else ()

    find_package (BLAS REQUIRED)
    find_package (LAPACK REQUIRED)

    add_library (asgard::LINALG INTERFACE IMPORTED)
    target_link_libraries (asgard::LINALG INTERFACE BLAS::BLAS LAPACK::LAPACK)

    # looking for the required cblas header, note that while BLAS is needed at runtime
    # the header is private and needed only to build, thus the path is not exported during install
    # on an Apple system, search for "Accelerate/Accelerate.h", assume Apple handles that
    # on an MKL system, search for mkl_cblas.h, assume already set in MKL environment
    # otherwise search for cblas.h (e.g., using OpenBLAS)
    if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(ASGARD_USING_APPLEBLAS ON)
    else()
        string(FIND "${BLAS_LIBRARIES}" "mkl" __asgard_mkl_pos)

        if (__asgard_mkl_pos GREATER_EQUAL 0)
            set(ASGARD_USING_MKL ON)
        else()
            find_path(__asg_check_cblas cblas.h)
            if (NOT __asg_check_cblas)
                # some OpenBLAS installs put cblas.h in a sub-folder
                find_path(__asg_cblas "cblas.h" PATH_SUFFIXES "openblas")
                if (NOT __asg_cblas)
                    message(FATAL_ERROR
"ASGarD found a BLAS package but could not find the required cblas.h header \
please include the path to the header in CMAKE_INCLUDE_PATH, e.g., -DCMAKE_INCLUDE_PATH=/some/path/ \
or set environment variable CMAKE_INCLUDE_PATH")
                endif()
                target_include_directories(asgard::LINALG INTERFACE ${__asg_cblas})
            else()
                target_include_directories(asgard::LINALG INTERFACE ${__asg_check_cblas})
            endif()
        endif()
        unset(__asgard_mkl_pos)
    endif()
endif ()
