#pragma once

namespace asgard
{

template<typename P>
struct linear_basis
{
  static P constexpr s3 = 1.73205080756887729; // sqrt(3.0)
  // projection basis
  static P pleg0(P) { return 1; }
  static P pleg1(P x) { return 2 * s3 * x - s3; }
  static P pwav0L(P x) { return s3 * (1 - 4 * x); }
  static P pwav0R(P x) { return s3 * (-3 + 4 * x); }
  static P pwav1L(P x) { return -1 + 6 * x; }
  static P pwav1R(P x) { return -5 + 6 * x; }
  // combine the projection basis together, maybe too much if-statements
  static P pbasis(int global_index, int cell_index, P x)
  {
    if (global_index == 0) // Legendre cell
      return (cell_index == 0) ? 1 : pleg1(x);
    else // wavelet cell
      if (cell_index == 0)
        return (x < 0.5) ? pwav0L(x) : pwav0R(x);
      else
        return (x < 0.5) ? pwav1L(x) : pwav1R(x);
  }
  // interpolation basis
  static P ibas0(P x) { return -3 * x + 2; }
  static P ibas1(P x) { return 3 * x - 1; }
  static P iwav0L(P x) { return -6 * x + 2; }
  static P iwav0R(P) { return 0; }
  static P iwav1L(P) { return 0; }
  static P iwav1R(P x) { return 6 * x - 4; }
};


} // namespace asgard
