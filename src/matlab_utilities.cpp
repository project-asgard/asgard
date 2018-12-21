
#include "matlab_utilities.hpp"
#include "tensors.hpp"

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
//
// these binary files can be generated from matlab or octave with
//
// function writeToFile(path, toWrite)
// fd = fopen(path,'w');
// fwrite(fd,toWrite,'double');
// fclose(fd);
// end
//
//-----------------------------------------------------------------------------
fk::vector<double> readVectorFromBinFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in | std::ios::binary);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::streampos bytes;

  // Get size, seek back to beginning
  infile.seekg(0, std::ios::end);
  bytes = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // create output vector
  fk::vector<double> values;

  unsigned int const num_values = bytes / sizeof(double);
  values.resize(num_values);

  infile.read(reinterpret_cast<char *>(values.data()), bytes);

  return values;

  // infile implicitly closed on exit
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// w = linspace(-1,1);
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::vector<double> readVectorFromTxtFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "type:");
  infile >> tmp_str;
  assert(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // make sure we're working with a (either row or column) vector
  assert((rows == 1) || columns == 1);

  int const num_elems = (rows >= columns) ? rows : columns;

  // create output vector
  fk::vector<double> values;

  values.resize(num_elems);

  for (auto i = 0; i < num_elems; ++i)
  {
    infile >> values(i);
  }

  return values;
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// m = rand(3,3)
// save outfile.dat m
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::matrix<double> readMatrixFromTxtFile(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    return {};
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "type:");
  infile >> tmp_str;
  assert(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  assert(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // create output matrix
  fk::matrix<double> values(rows, columns);

  for (auto i = 0; i < rows; ++i)
    for (auto j = 0; j < columns; ++j)
    {
      infile >> tmp_str;
      values(i, j) = std::stod(tmp_str);
    }

  return values;
}
