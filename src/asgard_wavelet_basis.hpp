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
  // interpolation basis
  static P ibas0(P x) { return -3 * x + 2; }
  static P ibas1(P x) { return 3 * x - 1; }
  static P iwav0L(P x) { return -6 * x + 2; }
  static P iwav0R(P) { return 0; }
  static P iwav1L(P) { return 0; }
  static P iwav1R(P x) { return 6 * x - 4; }
};

template<typename P>
struct quadratic_basis
{
  static P constexpr s3 = linear_basis<P>::s3; // sqrt(3.0)
  static P constexpr s5 = 2.2360679774997897;  // sqrt(5.0)

  static constexpr auto pleg0 = linear_basis<P>::pleg0;
  static constexpr auto pleg1 = linear_basis<P>::pleg1;
  static P pleg2(P x) { return s5 * (6.0 * x * x - 6.0 * x + 1.0); }

  static P pwav0L(P x) { return -(1.0 - 12.0 * x + 24.0 * x * x) * s5; }
  static P pwav1L(P x) { return s3 * (30.0 * x * x - 14.0 * x + 1.0); }
  static P pwav2L(P x) { return 1.0 - 6.0 * x; }

  static P pwav0R(P x) { return (13.0 - 36.0 * x + 24.0 * x * x) * s5; }
  static P pwav1R(P x) { return s3 * (30.0 * x * x - 46.0 * x + 17.0); }
  static P pwav2R(P x) { return 5.0 - 6.0 * x; }
};

} // namespace asgard
