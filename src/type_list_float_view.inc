#ifdef ASGARD_USE_DOUBLE_PREC
X(double, device)
X(double, host)
#else
#ifdef ASGARD_USE_LONG_DOUBLE
X(long double, host)
X(long double, device)
#else
X(float, host)
X(float, device)
#endif
#endif