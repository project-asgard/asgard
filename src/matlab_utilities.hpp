#pragma once

#include "fast_math.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace asgard
{
//-----------------------------------------------------------------------------
//
// a collection of utility functions to make working with matlab-like
// code a little easier
//
// - matlab functions ported to c++
//    - linspace (scalar inputs only)
//
// - matlab/octave file IO
//    - read_vector_from_txt_file (tested for octave)
//    - read_vector_from_bin_file (tested for octave and matlab)
//
//-----------------------------------------------------------------------------

// matlab's "linspace(start, end, N)" function
//-----------------------------------------------------------------------------
//
// c++ implementation of matlab (a subset of) linspace() function
// initial c++ implementation by Tyler McDaniel
//
// -- linspace (START, END)
// -- linspace (START, END, N)
//     Return a row vector with N linearly spaced elements between START
//     and END.
//
//     If the number of elements is greater than one, then the endpoints
//     START and END are always included in the range.  If START is
//     greater than END, the elements are stored in decreasing order.  If
//     the number of points is not specified, a value of 100 is used.
//
//     The 'linspace' function returns a row vector when both START and
//     END are scalars.
//
//  (unsupported)
//     If one, or both, inputs are vectors, then
//     'linspace' transforms them to column vectors and returns a matrix
//     where each row is an independent sequence between
//     'START(ROW_N), END(ROW_N)'.
//
//     For compatibility with MATLAB, return the second argument (END)
//     when only a single value (N = 1) is requested.
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

// norm( , 2) function, only for real vectors (2-norm)
template<typename P>
P l2_norm(fk::vector<P> const &vec);

// norm( , 'inf') function, only for real vectors (infinity-norm)
template<typename P>
P inf_norm(fk::vector<P> const &vec);

// find the indices in an fk::vector for which the predicate is true
template<typename P, typename Func>
fk::vector<int> find(fk::vector<P> const vect, Func pred)
{
  auto iter = vect.begin();
  std::vector<int> result;
  while ((iter = std::find_if(iter, vect.end(), pred)) != vect.end())
  {
    result.push_back(std::distance(vect.begin(), iter++));
  }
  return fk::vector<int>(result);
}
// find for a matrix. returns a two-column matrix
// whose rows are (r, c) indices satisfying the predicate
template<typename P, typename Func>
fk::matrix<int> find(fk::matrix<P> const matrix, Func pred)
{
  auto iter    = matrix.begin();
  int num_rows = matrix.nrows();

  std::vector<int> result_rows;
  std::vector<int> result_cols;

  while ((iter = std::find_if(iter, matrix.end(), pred)) != matrix.end())
  {
    int const index = std::distance(matrix.begin(), iter++);
    result_rows.push_back(index % num_rows);
    result_cols.push_back(index / num_rows);
  }
  int const num_entries = result_rows.size();
  if (num_entries == 0)
  {
    return fk::matrix<int>();
  }
  fk::matrix<int> result(num_entries, 2);
  result.update_col(0, result_rows);
  result.update_col(1, result_cols);

  return result;
}

// read a matlab vector from binary file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> read_vector_from_bin_file(std::filesystem::path const &path);

// read an octave double from text file
double read_scalar_from_txt_file(std::filesystem::path const &path);

// read an octave vector from text file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
fk::vector<double> read_vector_from_txt_file(std::filesystem::path const &path);

// read an octave matrix from text file into a fk::matrix
fk::matrix<double> read_matrix_from_txt_file(std::filesystem::path const &path);

// stitch matrices having equal # of rows together horizontally
template<typename P>
fk::matrix<P> horz_matrix_concat(std::vector<fk::matrix<P>> const matrices);

// limited subset of matbal meshgrid
fk::matrix<int> meshgrid(int const start, int const length);

template<typename P, mem_type mem>
fk::matrix<P> reshape(fk::matrix<P, mem> mat, int const nrow, int const ncol);

template<typename P>
fk::vector<P>
interp1(fk::vector<P> sample, fk::vector<P> values, fk::vector<P> coords);
} // namespace asgard