#pragma once

#include "asgard_tools.hpp"

/*!
 * \defgroup asgard_compute ASGarD Accelerated computing algorithms
 *
 * Tools for accelerating CPU and GPU computing.
 */

////////////////////////////////////////////////////////////////////////////////
//    OpenMP section: macros for calling OpenMP parallel and simd
////////////////////////////////////////////////////////////////////////////////
// As of LLVM version 18, clang does not utilize #pragma omp simd
// resulting in under-performing code, the macros disable omp simd directives
// when suing the clang compiler
// Use ASGARD_OMP_SIMD, ASGARD_OMP_PARFOR_SIMD or variants with extra options
#define ASGARD_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define ASGARD_PRAGMA_OMP_SIMD(x)
#define ASGARD_OMP_SIMD
#define ASGARD_OMP_PARFOR_SIMD
#define ASGARD_OMP_PARFOR_SIMD_EXTRA(x)
#else
#define ASGARD_OMP_SIMD ASGARD_PRAGMA(omp simd)
#define ASGARD_PRAGMA_OMP_SIMD(clause) ASGARD_PRAGMA(omp simd clause)
#define ASGARD_OMP_PARFOR_SIMD ASGARD_PRAGMA(omp parallel for simd)
#define ASGARD_OMP_PARFOR_SIMD_EXTRA(clause) ASGARD_PRAGMA(omp parallel for simd clause)
#endif

namespace asgard
{

}
