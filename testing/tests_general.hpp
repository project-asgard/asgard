//-----------------------------------------------------------------------------
//
// some utilities possibly useful/needed across various component tests
//
//-----------------------------------------------------------------------------

#ifndef _tests_general_h_
#define _tests_general_h_

#include "catch.hpp"
#include <vector>

void compareVectors(std::vector<double> a, std::vector<double> b);
void compare2dVectors(std::vector<std::vector<double>> a,
                      std::vector<std::vector<double>> b);

#endif
