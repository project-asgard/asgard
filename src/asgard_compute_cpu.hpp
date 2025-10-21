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

/*!
 * \brief Default precision to use, double if enabled and float otherwise.
 */
#ifdef ASGARD_ENABLE_DOUBLE
using default_precision = double;
#else
using default_precision = float;
#endif

/*!
 * \brief Indicated if computing should be done suing the CPU or GPU.
 *
 * This allows differentiating the array modes for the inputs into a function.
 */
enum class compute_mode {
  //! Using the CPU device
  cpu,
  //! Using the GPU device
  gpu
};

/*!
 * \brief Common namespace for defining various GPU utilities
 */
namespace gpu {
// the devise is here so that it is not copied in all GPU headers
/*!
 * \brief Strong type to identify the GPU device ID.
 */
struct device {
  //! Make a new device identifier
  explicit device(int gpuid) : id(gpuid) {}
  //! Compare two devices and if they match
  bool operator == (device const &other) const { return (id == other.id); }
  //! The device ID, e.g., 0, 1, 2, 3, ...
  int id = -1; // default to an invalid ID, forces an error if used uninitialized
};

}

/*!
 * \brief Math utilities for commonly used operations
 *
 * Many multi-index operations require the use of methods such as log()
 * and pow(), but use integer arithmetic instead.
 * This namesapce provides shorthand operations for methods that
 * compute the power of 2, power with integer component, integer log-2,
 * and several others.
 */
namespace fm {
//! computes 2^exponent using bit-shift operations, only for int-like types
template<typename T>
inline constexpr T ipow2(T const exponent)
{
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned> ||
                std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
                std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned long long>);
  expect(exponent >= 0);
  expect(exponent < std::numeric_limits<T>::digits);
  return T{1} << exponent;
}

//! Raise the base to an integer power
template<typename T = int64_t>
inline constexpr T ipow(T base, int exponent)
{
  expect(exponent >= 1);
  T result = base;
  for (int e = 1; e < exponent; e++)
    result *= base;
  return result;
}

//! computes std::floor( std::log2(x) ), returns 0 for x = 0 using bit-wise shifts
inline constexpr int intlog2(int x)
{
  int result = 0;
  while (x >>= 1)
    result++;
  return result;
}
//! computes std::pow( 2, std::floor( std::log2(x) ) ) using bit-wise shifts
inline int ipow2_log2(int x)
{
  int result = 1;
  while (x >>= 1)
    result <<= 1;
  return result;
}
//! computes ipow2_log2(i) and std::pow(std::sqrt(2.0), intlog2(i))
inline void intlog2_pow2pows2(int x, int &i2l2, double &is2l2)
{
  i2l2  = 1;
  is2l2 = 1.0;
  while (x >>= 1)
  {
    i2l2 <<= 1;
    is2l2 *= 1.41421356237309505; // sqrt(2.0)
  }
}
//! computes base^p where p is in integer
template<typename P>
P powi(P base, int p) {
  P res = 1;
  while (--p > -1)
    res *= base;
  return res;
}

/*!
 * \brief Computes the l-inf norm of the difference between x and y
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto diff_inf(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == static_cast<index>(y.size()));

  precision m{0};
  for (index i = index{0}; i < x.size(); i++)
    m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

//! \brief returns the max norm of an array
template<typename P>
P nrm_inf(int n, P const x[]) {
  P r = 0;
  for (int i = 0; i < n; i++) r = std::max(r, std::abs(x[i]));
  return r;
}

/*!
 * \brief Computes the root-mean-square-error between two vectors
 *
 * This works with all std::vector, std::array and fk::vector.
 * Does not work with GPU vectors and does not check if the data is on the device.
 */
template<typename vecx, typename vecy>
auto rmserr(vecx const &x, vecy const &y)
{
  using precision = typename vecx::value_type;
  using index     = decltype(x.size());
  expect(x.size() == y.size());

  precision err{0};
  for (index i = index{0}; i < x.size(); i++)
  {
    precision const d = x[i] - y[i];
    err += d * d;
  }
  return std::sqrt(err / x.size());
}

} // namespace fm

#ifdef ASGARD_USE_MPI
/*!
 * \brief Optional call to library initialization
 *
 * The ASGarD library itself does not require initialization; however, ASGarD may be using
 * components that require initialization, e.g., MPI.
 * The final executable can initialize MPI in one of two ways:
 * \code
 *   #include "asgard.hpp"
 *
 *   int main(int argc, char **argv) {
 *     asgard::libasgard_init(argc, argv);
 *     ...
 *     asgard::libasgard_finish();
 * \endcode
 * The methods will call the proper initialization methods, if MPI has been enabled,
 * otherwise the methods will do nothing.
 *
 * Alternatively, MPI_Init can be called directly
 * \code
 *   #include "asgard.hpp"
 *
 *   int main(int argc, char **argv) {
 *     #ifdef ASGARD_USE_MPI
 *     MPI_Init(&argc, &argv);
 *     #endif
 *     ...
 *     #ifdef ASGARD_USE_MPI
 *     MPI_Finalize();
 *     #endif
 * \endcode
 *
 * Naturally, the #ifdef directives can be omitted or the initialization can be skipped
 * altogether, if the given PDE definition always or never uses MPI.
 */
inline void libasgard_init(int &argc, char **&argv) {
  MPI_Init(&argc, &argv);
}
//! finalization of the library, see asgard::libasgard_init
inline void libasgard_finish() {
  MPI_Finalize();
}
//! RAII style call to init/finish
struct libasgard_runtime {
  //! calls libasgard_init
  libasgard_runtime(int &argc, char **&argv) {
    libasgard_init(argc, argv);
  }
  //! calls libasgard_finish
  ~libasgard_runtime() { libasgard_finish(); }
};
#else
inline void libasgard_init(int &, char **&) {}
inline void libasgard_finish() {}
struct libasgard_runtime {
  //! does nothing
  libasgard_runtime(int &, char **&) {}
};
#endif

}
