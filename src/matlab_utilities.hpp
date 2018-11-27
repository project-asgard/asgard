
//-----------------------------------------------------------------------------
//
// a collection of utility functions to make working with matlab-like
// code a little easier
//
// - matlab functions ported to c++
//    - linspace (scalar inputs only)
//
// - matlab/octave file IO
//    - readVectorFromTxtFile (tested for octave)
//    - readVectorFromBinFile (tested for octave and matlab)
//
//-----------------------------------------------------------------------------

#ifndef _matlab_utilities_h_
#define _matlab_utilities_h_

#include "tensors.hpp"
#include <string>
#include <vector>

// matlab's "linspace(start, end, N)" function
std::vector<double> linspace(double const start, double const end,
                             unsigned int const num_elems = 100);

// matlab's "eye(M, N)" function
fk::matrix eye(int const M = 1);
fk::matrix eye(int const M, int const N);

// read a matlab vector from binary file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
std::vector<double> readVectorFromBinFile(std::string const &path);

// read an octave vector from text file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
std::vector<double> readVectorFromTxtFile(std::string const &path);

// read an octave matrix from text file into a fk::matrix
namespace fk
{
matrix readMatrixFromTxtFile(std::string const &path);
}

#endif
