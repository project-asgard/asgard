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
#  Setup a menu of BLAS/LAPACK venders.
#-------------------------------------------------------------------------------
set (BLA_VENDOR All CACHE STRING "BLAS/LAPACK Vendor")
set_property (CACHE BLA_VENDOR PROPERTY STRINGS
              All
              ACML ACML_MP ACML_GPU
              Apple NAS
              Arm Arm_mp Arm_ilp64 Arm_ilp64_mp
              ATLAS
              FLAME
              FlexiBLAS
              Fujitsu_SSL2 Fujitsu_SSL2BLAMP
              IBMESSL
              Intel10_32 Intel10_64lp Intel10_64lp_seq Intel10_64ilp Intel10_64ilp_seq Intel10_64_dyn
              NVHPC
              OpenBLAS
              SCSL
)

#  Check for platform provided BLAS and LAPACK libaries. If these were not found
#  then build the openblas library.
if (NOT ${ASGARD_BUILD_OPENBLAS})
    find_package (BLAS)
    find_package (LAPACK)

    if (NOT ${BLAS_FOUND} OR NOT ${LAPACK_FOUND})

#  Set the ASGARD_BUILD_OPENBLAS option to true in the cmake gui since we now
#  need to build OpenBLAS.
        set (ASGARD_BUILD_OPENBLAS ON CACHE BOOL "Download and build our own OpenBLAS" FORCE)

    endif ()
endif ()

#-------------------------------------------------------------------------------
#  Setup and build OpenBLAS if ASGARD_BUILD_OPENBLAS is ON
#-------------------------------------------------------------------------------
if (${ASGARD_BUILD_OPENBLAS})
    register_project (openblas
                      OPENBLAS
                      https://github.com/xianyi/OpenBLAS.git
                      v0.3.18
                      ON
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
    endforeach ()

    set (BLAS_FOUND 1)
    set (LAPACK_FOUND 1)
    find_package_handle_standard_args (LINALG
                                       REQUIRED_VARS BLAS_FOUND LAPACK_FOUND)

    add_library (LINALG::LINALG INTERFACE IMPORTED)
    target_link_libraries (LINALG::LINALG
                           INTERFACE
                           openblas
    )

#  Manually set the openblas include directory since openblas only sets the
#  include directory for the install.
    target_include_directories (LINALG::LINALG
                                INTERFACE
                                ${FETCHCONTENT_BASE_DIR}/openblas-build
    )

    target_compile_definitions (LINALG::LINALG
                                INTERFACE
                                ASGARD_OPENBLAS
    )
else ()
    find_package_handle_standard_args (LINALG
                                       REQUIRED_VARS BLAS_FOUND LAPACK_FOUND)

    add_library (LINALG::LINALG INTERFACE IMPORTED)
    target_link_libraries (LINALG::LINALG
                           INTERFACE
                           $<$<BOOL:${BLAS_FOUND}>:BLAS::BLAS>
                           $<$<BOOL:${LAPACK_FOUND}>:LAPACK::LAPACK>
    )

    target_compile_definitions (LINALG::LINALG
                                INTERFACE
                                $<$<OR:$<AND:$<PLATFORM_ID:Darwin>,$<STREQUAL:${BLA_VENDOR},All>>,$<STREQUAL:${BLA_VENDOR},Apple>,$<STREQUAL:${BLA_VENDOR},NAS>>:ASGARD_ACCELERATE>
                                $<$<STREQUAL:${BLA_VENDOR},OpenBLAS>:ASGARD_OPENBLAS>
    )
endif ()
