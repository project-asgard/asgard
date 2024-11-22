#pragma once
#include "asgard_fast_math.hpp"

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

//! returns a dense identity matrix
template<typename P>
fk::matrix<P> eye(int const M)
{
  fk::matrix<P> id(M, M);
  for (int i = 0; i < M; ++i)
    id(i, i) = 1.0;
  return id;
}

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
fk::matrix<int> find(fk::matrix<P> const &matrix, Func pred)
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
template<typename P>
fk::vector<P> read_vector_from_bin_file(std::filesystem::path const &path);

// read an octave double from text file
double read_scalar_from_txt_file(std::filesystem::path const &path);

// read an octave vector from text file into a std::vector
// note that fk::vector has a copy assignment overload from std::vector
template<typename P>
fk::vector<P> read_vector_from_txt_file(std::filesystem::path const &path);

// read an octave matrix from text file into a fk::matrix
template<typename P>
fk::matrix<P> read_matrix_from_txt_file(std::filesystem::path const &path);

// stitch matrices having equal # of rows together horizontally
template<typename P>
fk::matrix<P> horz_matrix_concat(std::vector<fk::matrix<P>> const &matrices);

template<typename P>
fk::vector<P> interp1(fk::vector<P> const &sample, fk::vector<P> const &values,
                      fk::vector<P> const &coords);
} // namespace asgard
