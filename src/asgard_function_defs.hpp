#pragma once

#include "asgard_momentset.hpp"

/*!
 * \defgroup asgard_funcdef ASGarD Function Definitions
 *
 * \par Functions
 * The PDE term coefficients, sources, initial and boundary conditions must be defined
 * by functions, e.g., y = f(x).
 * ASGarD uses std::function with many different signatures to handle different cases.
 * The std::function approach allows ASGarD to be compiled as a library and work with
 * user provided definitions after the installation, sometimes this is referred to as
 * having an "open" or "extensible" set of types and functionality.
 * The C++ std::function uses v-tables and polymorphic jumps with performance overhead
 * and is mitigated by using "batch" calls, e.g., calling f(x) for a set (vector) of points,
 * as opposed to making a separate v-table jump for each quadrature point
 * in each finite element cell.
 *
 * \par Inputs and const correctness
 * Most of the functions signatures use std::vector with either float or double precision
 * user provided functions should \b never resize the vectors or violate const-correctness,
 * e.g., by modifying the entries of vectors marked as "const".
 * All vectors and arrays will be pre-allocated to the correct size.
 *
 * \par Naming conventions
 * Function type names that start with "s" relate to a single-dimensional or scalar context,
 * while "md" indicates multidimensional context, i.e., all dimensions defined in
 * the problem and set in the asgard::pde_domain.
 * Suffix "_f" indicates an additional input field, e.g., F(x, y) vs. F(x, y, f(x, y)).
 * Having "mom" in the name indicates moment dependence and the asgard::momentset will be passed
 * into all function calls.
 * The "gpu" indicates that all arrays/pointers relate to data on the GPU device.
 *
 * \par GPU context
 * The GPU context requires either CUDA or ROCM to be enabled in ASGarD.
 * The function signatures use "raw arrays", e.g., "double x[]", as to avoid forcing the use
 * of custom containers.
 *
 */

namespace asgard
{
/*!
 * \ingroup asgard_funcdef
 * \brief Scalar function, returning y = F(t)
 *
 * The most basic function and the only one that doesn't use batch computing and has non-void
 * return type.
 * The function is used as the time component of asgard::separable_func.
 */
template<typename P>
using scalar_func = std::function<P(P const t)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Vector function, computing fx = F(x), no time-dependence
 *
 * Represents a single-dimensional function that is "fixed" in time.
 *
 * On entry, \b x is a set of quadrature points within the domain of some direction.
 *
 * On exit, \b fx must contain the corresponding values, e.g., x[i] corresponds to fx[i].
 */
template<typename P>
using sfixed_func1d = std::function<void(std::vector<P> const &x, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Vector function, computing fx = F(x, t)
 *
 * Represents a single-dimensional function that varies in time.
 *
 * The signature is the same as asgard::sfixed_func1d with the additional time component.
 */
template<typename P>
using svector_func1d = std::function<void(std::vector<P> const &x, P t, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Vector function, computing fx = F(x, f), where f is a field, e.g., moment of the solution
 *
 * Represents a single-dimensional function that is "fixed" in time and accepts an additional
 * field parameter.
 *
 * On entry, the three vectors will have the same size, \b x will contain the quadrature
 * point values, \b m will have the additional field values.
 *
 * On exit, \b fx[i] must have the value corresponding to x[i] and m[i]
 */
template<typename P>
using sfixed_func1d_f = std::function<void(std::vector<P> const &x, std::vector<P> const &m, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a non-separable function, fx = F(t, x)
 *
 * Represents a multidimensional function with time-dependence in \b t,
 * this can be used for either source or initial condition.
 *
 * On entry, \b x contains the points so that the i-th point is at (x[i][0], x[i][1], ..., x[i][d])
 * where d is the number of dimensions and it is equal to x.stride(), see asgard::vector2d
 *
 * Similarly, \b fx will have size equal to x.num_strips()
 *
 * On exit, \b fx[i] must have the values corresponding to x[i][0] ... x[i][d]
 */
template<typename P>
using md_func = std::function<void(P t, vector2d<P> const &x, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a non-separable function that accepts an additional field parameter
 *
 * The signature is the same as asgard::md_func but has an additional field parameter \b f
 * with size matching x.num_strips() and fx.size()
 *
 * The function is used to define term coefficient, where f will be values of the field that
 * the term is acting on.
 */
template<typename P>
using md_func_f = std::function<void(P t, vector2d<P> const &x, std::vector<P> const &f, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a non-separable function with moment dependence
 *
 * The signature is the same as asgard::md_func but has an additional moment dependence.
 *
 * Moments are created by asgard::pde_scheme::register_moment which returns an asgard::moment_id,
 * using the id the vector with values for the moments can be accessed with moment[id].
 *
 * See \ref asgard_examples_bgk "BGK example" for details.
 */
template<typename P>
using md_mom_func = std::function<void(P t, vector2d<P> const &x, momentset<P> const &moments, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a non-separable function with field and moment parameters
 *
 * The signature is the same as asgard::md_func but has an additional field parameter \b f,
 * similar to the other "_f" functions.
 */
template<typename P>
using md_mom_func_f = std::function<void(P t, vector2d<P> const &x, momentset<P> const &moments, std::vector<P> const &f, std::vector<P> &vals)>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a GPU non-separable function
 *
 * All "gpu" functions require either CUDA or ROCM support and the pointers will reference data
 * on the GPU device.
 *
 * On entry, \b num is the number of points and the size of fx, the number of needed since the raw
 * pointers don't have .size()
 *
 * The length of \b x is num * num-dims and the i-th point starts from x[i * num_dims].
 */
template<typename P>
using md_gpu_func = std::function<void(int64_t const num, P t, P const x[], P fx[])>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a GPU non-separable function that accepts an additional field parameter
 *
 * This is a GPU version of asgard::md_func_f, where \b num specifies the size of \b x, \b f
 * and \b fx, while \b x is as in asgard::md_gpu_func
 */
template<typename P>
using md_gpu_func_f = std::function<void(int64_t const num, P t, P const x[], P const f[], P fx[])>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a non-separable function with moment dependence on the GPU
 *
 * This is the GPU version of asgard::md_mom_func with inputs similar to asgard::md_gpu_func
 *
 * The \b moments work similar to the CPU context but the returned moments are gpu::vectors
 * that hold .data() on the GPU device.
 */
template<typename P>
using md_gpu_mom_func = std::function<void(int64_t const num, P t, P const x[], momentset_gpu<P> const &moments, P fx[])>;

/*!
 * \ingroup asgard_funcdef
 * \brief Signature for a GPU non-separable function that accepts an moment and field parameters
 *
 * This is the GPU version of asgard::md_mom_func_f with inputs similar to asgard::md_gpu_mom_func
 */
template<typename P>
using md_gpu_mom_func_f = std::function<void(int64_t const num, P t, P const x[], momentset_gpu<P> const &moments, P const f[], P fx[])>;

#ifndef __ASGARD_DOXYGEN_SKIP
//! variant holding any of the possible multidimensional source functions
template<typename P>
using md_source_func = std::variant<std::monostate, md_func<P>, md_mom_func<P>, md_gpu_func<P>, md_gpu_mom_func<P>>;
//! variant holding any of the possible multidimensional field functions
template<typename P>
using md_field_func = std::variant<std::monostate, md_func_f<P>, md_mom_func_f<P>, md_gpu_func_f<P>, md_gpu_mom_func_f<P>>;

//! trait type that indicates if a function signature uses moments
template<typename F> struct uses_mom_trait : std::false_type {};
//! specializations
template<typename P> struct uses_mom_trait<md_mom_func<P>> : std::true_type {};
template<typename P> struct uses_mom_trait<md_mom_func_f<P>> : std::true_type {};
template<typename P> struct uses_mom_trait<md_gpu_mom_func<P>> : std::true_type {};
template<typename P> struct uses_mom_trait<md_gpu_mom_func_f<P>> : std::true_type {};

template<typename F> constexpr bool uses_moments = uses_mom_trait<F>::value;

//! trait type indicating if the GPU is being used
template<typename F> struct uses_gpu_trait : std::false_type {};
//! specializations
template<typename P> struct uses_gpu_trait<md_gpu_func<P>> : std::true_type {};
template<typename P> struct uses_gpu_trait<md_gpu_func_f<P>> : std::true_type {};
template<typename P> struct uses_gpu_trait<md_gpu_mom_func<P>> : std::true_type {};
template<typename P> struct uses_gpu_trait<md_gpu_mom_func_f<P>> : std::true_type {};

template<typename F> constexpr bool uses_gpu = uses_gpu_trait<F>::value;

template<typename> constexpr bool is_valid_call = false;

#ifdef ASGARD_USE_GPU
template<typename> constexpr bool has_gpu_enabled = true;
#else
template<typename> constexpr bool has_gpu_enabled = false;
#endif

#endif

}
