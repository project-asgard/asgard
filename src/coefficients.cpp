#include "coefficients.hpp"
#include "pde.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"

// construct 1D coefficient matrix
// this routine returns a 2D array representing an operator coefficient
// matrix for a single dimension (1D). Each term in a PDE requires D many
// coefficient matricies.

template<typename P>
fk::matrix<P>
generate_coefficients(dimension<P> const dim,
                      std::vector<term<P>> const term_list, P const time)
{
  // setup jacobi of variable x and define coeff_mat
  int const two_to_level    = static_cast<int>(std::pow(2, dim.level));
  P const normalized_domain = (dim.domain_max - dim.domain_min) / two_to_level;
  int const degrees_freedom_1d = dim.degree * two_to_level;
  fk::matrix<P> coefficients(degrees_freedom_1d, degrees_freedom_1d);

  // set number of quatrature points (should this be order dependent?)
  // FIXME is this a global quantity??
  int const quad_num = 10;

  // get quadrature points and weights.
  auto const [roots, weights] = legendre_weights<P>(quad_num, -1.0, 1.0);

  // compute the trace values (values at the left and right of each element for
  // all k) trace_left is 1 by degree trace_right is 1 by degree
  // FIXME should these be vectors?
  fk::matrix<P> const trace_left =
      legendre<P>(fk::vector<P>({-1.0}), dim.degree)[0];
  fk::matrix<P> const trace_right =
      legendre<P>(fk::vector<P>({1.0}), dim.degree)[0];

  // get the basis functions and derivatives for all k
  auto const [legendre_poly, legendre_prime] = legendre(roots, dim.degree);

  // these matrices are quad_num by degree
  fk::matrix<P> basis = legendre_poly * (1.0 / std::sqrt(normalized_domain));
  fk::matrix<P> basis_prime =
      legendre_prime *
      (1.0 / std::sqrt(normalized_domain) * 2.0 / normalized_domain);

  // Mass = sparse(dof_1D,dof_1D);
  // Grad = sparse(dof_1D,dof_1D);
  // Stif = sparse(dof_1D,dof_1D);
  // Flux = sparse(dof_1D,dof_1D);
  return fk::matrix<P>();
}

template fk::matrix<float>
generate_coefficients(dimension<float> const dim,
                      std::vector<term<float>> const term_list,
                      float const time);

template fk::matrix<double>
generate_coefficients(dimension<double> const dim,
                      std::vector<term<double>> const term_list,
                      double const time);
