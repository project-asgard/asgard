#include "blas_free.hpp"

// axpy - add the argument vector scaled by alpha
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &
axpy(P const alpha, fk::vector<P, omem> const &x, fk::vector<P, mem> &y)
{
  assert(x.size() == y.size());
  int n    = x.size();
  int one  = 1;
  P alpha_ = alpha;

  if constexpr (std::is_same<P, double>::value)
  {
    daxpy_(&n, &alpha_, x.data(), &one, y.data(), &one);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    saxpy_(&n, &alpha_, x.data(), &one, y.data(), &one);
  }
  else
  {
    for (auto i = 0; i < x.size(); ++i)
    {
      y(i) = y(i) + x(i) * alpha_;
    }
  }

  return y;
}

// copy(x,y) - copy vector x into y
template<typename P, mem_type mem, mem_type omem>
fk::vector<P, mem> &copy(fk::vector<P, omem> const &x, fk::vector<P, mem> &y)
{
  assert(x.size() == y.size());
  int n   = x.size();
  int one = 1;

  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(&n, x.data(), &one, y.data(), &one);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    scopy_(&n, x.data(), &one, y.data(), &one);
  }
  else
  {
    for (auto i = 0; i < x.size(); ++i)
    {
      y(i) = x(i);
    }
  }

  return y;
}

// scal - scale a vector
template<typename P, mem_type mem>
fk::vector<P, mem> &scal(P const alpha, fk::vector<P, mem> &x)
{
  int one_i = 1;
  int n     = x.size();
  P alpha_  = alpha;

  if constexpr (std::is_same<P, double>::value)
  {
    dscal_(&n, &alpha_, x.data(), &one_i);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sscal_(&n, &alpha_, x.data(), &one_i);
  }
  else
  {
    for (int i = 0; i < n; ++i)
    {
      x(i) = x(i) * alpha_;
    }
  }
  return x;
}

template fk::vector<float, mem_type::owner> &
axpy(float const alpha, fk::vector<float, mem_type::owner> const &x,
     fk::vector<float, mem_type::owner> &y);
template fk::vector<double, mem_type::owner> &
axpy(double const alpha, fk::vector<double, mem_type::owner> const &x,
     fk::vector<double, mem_type::owner> &y);
template fk::vector<int, mem_type::owner> &
axpy(int const alpha, fk::vector<int, mem_type::owner> const &x,
     fk::vector<int, mem_type::owner> &y);

template fk::vector<float, mem_type::owner> &
axpy(float const alpha, fk::vector<float, mem_type::view> const &x,
     fk::vector<float, mem_type::owner> &y);
template fk::vector<double, mem_type::owner> &
axpy(double const alpha, fk::vector<double, mem_type::view> const &x,
     fk::vector<double, mem_type::owner> &y);
template fk::vector<int, mem_type::owner> &
axpy(int const alpha, fk::vector<int, mem_type::view> const &x,
     fk::vector<int, mem_type::owner> &y);

template fk::vector<float, mem_type::view> &
axpy(float const alpha, fk::vector<float, mem_type::owner> const &x,
     fk::vector<float, mem_type::view> &y);
template fk::vector<double, mem_type::view> &
axpy(double const alpha, fk::vector<double, mem_type::owner> const &x,
     fk::vector<double, mem_type::view> &y);
template fk::vector<int, mem_type::view> &
axpy(int const alpha, fk::vector<int, mem_type::owner> const &x,
     fk::vector<int, mem_type::view> &y);

template fk::vector<float, mem_type::view> &
axpy(float const alpha, fk::vector<float, mem_type::view> const &x,
     fk::vector<float, mem_type::view> &y);
template fk::vector<double, mem_type::view> &
axpy(double const alpha, fk::vector<double, mem_type::view> const &x,
     fk::vector<double, mem_type::view> &y);
template fk::vector<int, mem_type::view> &
axpy(int const alpha, fk::vector<int, mem_type::view> const &x,
     fk::vector<int, mem_type::view> &y);

template fk::vector<float, mem_type::owner> &
copy(fk::vector<float, mem_type::owner> const &x,
     fk::vector<float, mem_type::owner> &y);
template fk::vector<double, mem_type::owner> &
copy(fk::vector<double, mem_type::owner> const &x,
     fk::vector<double, mem_type::owner> &y);
template fk::vector<int, mem_type::owner> &
copy(fk::vector<int, mem_type::owner> const &x,
     fk::vector<int, mem_type::owner> &y);
template fk::vector<float, mem_type::owner> &
copy(fk::vector<float, mem_type::view> const &x,
     fk::vector<float, mem_type::owner> &y);
template fk::vector<double, mem_type::owner> &
copy(fk::vector<double, mem_type::view> const &x,
     fk::vector<double, mem_type::owner> &y);
template fk::vector<int, mem_type::owner> &
copy(fk::vector<int, mem_type::view> const &x,
     fk::vector<int, mem_type::owner> &y);

template fk::vector<float, mem_type::view> &
copy(fk::vector<float, mem_type::owner> const &x,
     fk::vector<float, mem_type::view> &y);
template fk::vector<double, mem_type::view> &
copy(fk::vector<double, mem_type::owner> const &x,
     fk::vector<double, mem_type::view> &y);
template fk::vector<int, mem_type::view> &
copy(fk::vector<int, mem_type::owner> const &x,
     fk::vector<int, mem_type::view> &y);
template fk::vector<float, mem_type::view> &
copy(fk::vector<float, mem_type::view> const &x,
     fk::vector<float, mem_type::view> &y);
template fk::vector<double, mem_type::view> &
copy(fk::vector<double, mem_type::view> const &x,
     fk::vector<double, mem_type::view> &y);
template fk::vector<int, mem_type::view> &
copy(fk::vector<int, mem_type::view> const &x,
     fk::vector<int, mem_type::view> &y);

template fk::vector<float, mem_type::owner> &
scal(float const alpha, fk::vector<float, mem_type::owner> &x);
template fk::vector<double, mem_type::owner> &
scal(double const alpha, fk::vector<double, mem_type::owner> &x);
template fk::vector<int, mem_type::owner> &
scal(int const alpha, fk::vector<int, mem_type::owner> &x);

template fk::vector<float, mem_type::view> &
scal(float const alpha, fk::vector<float, mem_type::view> &x);
template fk::vector<double, mem_type::view> &
scal(double const alpha, fk::vector<double, mem_type::view> &x);
template fk::vector<int, mem_type::view> &
scal(int const alpha, fk::vector<int, mem_type::view> &x);
