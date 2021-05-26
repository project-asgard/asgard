#ifdef ASGARD_USE_DOUBLE_PREC
X(double)
#else
#ifdef ASGARD_USE_LONG_DOUBLE
X(long double)
#else
X(float)
#endif
#endif
