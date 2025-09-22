#[==[
Find package that wraps functionality to find the BLAS/LAPACK libraries. By
default, it looks for the platform default. If that is not found, it will build
as openblas from source.

Provides the following variables:

  * `LINALG_FOUND`: Whether NetCDF was found or not.
  * `LINALG::LINALG`: A target to use with `target_link_libraries`.
#]==]

include (FindPackageHandleStandardArgs)

#-------------------------------------------------------------------------------
#  Setup and build OpenBLAS if ASGARD_BUILD_OPENBLAS is ON
#  otherwise use the CMake native find_package(BLAS)
#-------------------------------------------------------------------------------
if (ASGARD_BUILD_OPENBLAS)
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

    register_project (openblas
                      OPENBLAS
                      https://github.com/xianyi/OpenBLAS.git
                      v0.3.24
    )

#  Fetch content does not run the install phase so the headers for openblas are
#  not geting copied to the openblas-build directory. We will do this manually
#  instead.
    set (openblas_headers
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

    foreach (header IN LISTS openblas_headers)
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS/${header}
                        ${FETCHCONTENT_BASE_DIR}/openblas-build/${header}
                        COPYONLY)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/OPENBLAS/${header}
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

    string(FIND "${BLA_VENDOR}" "Intel" __asgard_intel_pos)

    if (__asgard_intel_pos GREATER_EQUAL 0)
        set(ASGARD_USING_MKL ON)
    endif()
    unset(__asgard_intel_pos)

endif ()
