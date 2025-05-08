#include "asgard_quadrature.hpp"

namespace asgard
{

std::array<std::vector<double>, 2>
legendre_vals(std::vector<double> const &points, int const degree,
              legendre_normalization const norm)
{
  expect(degree >= 0);
  expect(points.size() > 0);

  int const pdof = degree + 1;
  int const nump = static_cast<int>(points.size());

  // allocate and zero the output Legendre polynomials, their derivatives
  std::vector<double> vec_leg(points.size() * pdof);
  std::vector<double> vec_leg_prime(points.size() * pdof);

  span2d<double> leg(points.size(), pdof, vec_leg.data());
  span2d<double> leg_prime(points.size(), pdof, vec_leg_prime.data());

  std::fill_n(leg[0], nump, 1); // constant Legenre polynomial
  // vec_leg_prime is implicitly set to zero
  if (degree >= 1)
    std::copy(points.begin(), points.end(), leg[1]); // linear Legendre poly.

  // using the recurrence relation (k + 1) P_{k + 1} = (2k + 1) x P_k - k P_{k-1}
  for (int k = 1; k < degree; k++) {
    double const nscale = 2.0 * k + 1.0;
    double const dscale = 1.0 / (k + 1.0);

    for (int j = 0; j < nump; j++)
      leg[k+1][j] = points[j] * nscale * leg[k][j] * dscale;

    for (int j = 0; j < nump; j++)
      leg[k+1][j] -= leg[k-1][j] * k * dscale;
  }

  // vec_leg_prime[0] is already set to zero
  if (degree >= 1)
    std::fill_n(leg_prime[1], nump, 1); // derivative of linear is constant

  for (int k = 2; k <= degree; k++) {
    for (int j = 0; j < nump; j++)
      leg_prime[k][j] = k * (points[j] * leg[k][j] - leg[k - 1][j]) / (points[j] * points[j] - 1);
  }

  for (size_t j = 0; j < points.size(); j++) {
    if (points[j] < -1 or points[j] > 1) {
      for (int k = 0; k <= degree; k++) {
        leg[k][j] = 0;
        leg_prime[k][j] = 0;
      }
    }
  }

  if (degree > 0) // rescaling applies only to linears and above
  {
    switch (norm) {
      case legendre_normalization::lin:
        for (int k = 0; k <= degree; k++) {
          double const dscale = std::sqrt(2.0 * k + 1.0);
          for (int j = 0; j < nump; j++)
            leg[k][j] *= dscale;
          for (int j = 0; j < nump; j++)
            leg_prime[k][j] *= dscale;
        }
        break;
      case legendre_normalization::matlab:
        break;
      default: { // case legendre_normalization::unnormalized
          double const dscale = std::sqrt(2.0);
          for (size_t i = 0; i < vec_leg.size(); i++)
            vec_leg[i] *= dscale;
          for (size_t i = 0; i < vec_leg_prime.size(); i++)
            vec_leg_prime[i] *= dscale;
        }
        break;
    };
  }

  return {vec_leg, vec_leg_prime};
}

std::array<std::vector<double>, 2>
legendre_weights(int const degree, double const lower_bound, double const upper_bound,
                 quadrature_mode const quad_mode)
{
  expect(degree >= 0);
  expect(lower_bound < upper_bound);

  int const default_num_quad = std::max(ASGARD_NUM_QUADRATURE, degree + 2);

  int const num_points = (quad_mode == quadrature_mode::use_degree)
                        ? degree + 1
                        : default_num_quad;

  std::vector<double> x_roots(num_points);
  std::vector<double> weights(num_points);

  // Initial guess at the roots for the Legendre polynomial of degree num_points
  // x_roots = cos((2*(0:num_points-1)+1)*pi / (2*(num_points-1)+2))
  //          + (0.27/num_points) * sin(pi*x_linspace*((num_points-1)/(num_points+1);
  // The 0.27 is just a good guess that works well

  // reversing the order so the final set of point is ordered left-right
  // computing the cos() component
  {
    double const a = M_PI / (2.0 * num_points);
    for (int i = 0; i < num_points; i++)
      x_roots[num_points - i - 1] = std::cos((2.0 * i + 1.0) * a);
  }

  // computing the sin() component
  if (num_points > 1) {
    double const a = 0.27 / num_points;
    double const b = M_PI * (num_points - 1.0) / (num_points + 1.0);
    double const dx = 2.0 / (num_points - 1.0);
    double f = 0;
    for (int i = 1; i < num_points; i++) {
      x_roots[num_points - i - 1] += a * std::sin(b * f);
      f += dx;
    }
    x_roots.front() += a * std::sin(b);
  }

  // This piece of the code uses Newton's method to solve for the Legendre polynomial roots
  // x_roots = x_roots - f(x_roots) / f'(x_roots)
  // f() is the values of Legendre polynomials

  std::vector<double> workspace(4 * num_points);
  double *prev = workspace.data();
  double *curr = prev + num_points;
  double *next = curr + num_points;

  double *leg_prime = next + num_points;

  double diff = 1000; // make sure we enter the while loop below
  while (diff > std::numeric_limits<double>::epsilon())
  {
    // set the constant and linear polynomials, recurrence relation
    std::fill_n(prev, num_points, 1);
    std::copy_n(x_roots.begin(), num_points, curr);
    for (int i = 1; i < num_points; ++i)
    {
      // P_i+1(x_roots) = ((2*i+1)*x_roots*P_i(x_roots) - i*P_i-1(x_roots))/(i+1)
      double const nscale = 2.0 * i + 1.0;
      double const dscale = 1.0 / (i + 1.0);

      for (int j = 0; j < num_points; j++)
        next[j] = (x_roots[j] * nscale * curr[j] - prev[j] * i) * dscale;

      double *t = prev;
      prev = curr;
      curr = next;
      next = t;
    }

    diff = 0;
    for (int j = 0; j < num_points; j++)
    {
      // lp is the derivative of the Legenre polynomial
      double const lp = num_points * (prev[j] - curr[j] * x_roots[j]) / (1 - x_roots[j] * x_roots[j]);
      double const dl = curr[j] / lp; // Newton correction
      leg_prime[j] = lp;
      x_roots[j] = (lp * x_roots[j] - curr[j]) / lp;
      diff = std::max(diff, std::abs(dl));
    }
  }

  // Compute the weights
  for (int j = 0; j < num_points; j++)
    weights[j] = (upper_bound - lower_bound) / ((1 - x_roots[j] * x_roots[j]) * leg_prime[j] * leg_prime[j]);

  // remap to (lower, upper)
  for (int j = 0; j < num_points; j++)
    x_roots[j] = 0.5 * (lower_bound + upper_bound + (upper_bound - lower_bound) * x_roots[j]);

  return std::array<std::vector<double>, 2>{x_roots, weights};
}

} // namespace asgard
