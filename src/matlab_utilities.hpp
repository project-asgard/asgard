#pragma once

#include "tensors.hpp"
#include <string>

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

template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, fk::vector<P>>
linspace(P const start, P const end, unsigned int const num_elems = 100);

template<typename P>
fk::matrix<P> eye(int const M = 1);
template<typename P>
fk::matrix<P> eye(int const M, int const N);

template<typename P>
P polyval(fk::vector<P> const p, P const x);

template<typename P>
fk::vector<P> polyval(fk::vector<P> const p, fk::vector<P> const x);

// read a matlab vector from binary file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> readVectorFromBinFile(std::string const &path);

// read an octave vector from text file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> readVectorFromTxtFile(std::string const &path);

// read an octave matrix from text file into a fk::matrix
fk::matrix<double> readMatrixFromTxtFile(std::string const &path);

// suppress implicit instantiations
extern template fk::vector<float> linspace(float const start, float const end,
                                           unsigned int const num_elems = 100);
extern template fk::vector<double> linspace(double const start,
                                            double const end,
                                            unsigned int const num_elems = 100);

extern template fk::matrix<int> eye(int const M = 1);
extern template fk::matrix<float> eye(int const M = 1);
extern template fk::matrix<double> eye(int const M = 1);
extern template fk::matrix<int> eye(int const M, int const N);
extern template fk::matrix<float> eye(int const M, int const N);
extern template fk::matrix<double> eye(int const M, int const N);

extern template int polyval(fk::vector<int> const p, int const x);
extern template float polyval(fk::vector<float> const p, float const x);
extern template double polyval(fk::vector<double> const p, double const x);

extern template fk::vector<int>
polyval(fk::vector<int> const p, fk::vector<int> const x);
extern template fk::vector<float>
polyval(fk::vector<float> const p, fk::vector<float> const x);
extern template fk::vector<double>
polyval(fk::vector<double> const p, fk::vector<double> const x);
