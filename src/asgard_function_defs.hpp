#pragma once

#include "asgard_momentset.hpp"

/*!
 * \defgroup asgard_funcdef ASGarD Function Definitions
 *
 * The PDE term coefficients, sources, initial and boundary conditions, must be defined
 * by functions, e.g., y = f(x).
 * ASGarD uses std::function with many different signatures to handle different cases.
 * The C++ std::function uses v-tables and polymorphic jumps with performance overhead,
 * the std::function approach allows ASGarD to be compiled as a library and be open to
 * user provided definitions after the installation.
 * The performance hit is mitigated by using "batch" calls, e.g., calling f(x) for
 * a set of points, as opposed to making a separate call for each quadrature point.
 *
 * Most of the functions signatures use std::vector with either float or double precision,
 * several rules must be observed.
 * User provided functions should \b never resize the vectors or violate const-correctness,
 * e.g., by modifying the entries of vectors marked as "const".
 *
 *
 */

namespace asgard
{


}
