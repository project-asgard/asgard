#include "asgard_matlab_utilities.hpp"

namespace asgard
{
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
template<typename P>
fk::vector<P> read_vector_from_bin_file(std::filesystem::path const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in | std::ios::binary);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::streampos bytes;

  // Get size, seek back to beginning
  infile.seekg(0, std::ios::end);
  bytes = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // create output vector
  unsigned int const num_values = bytes / sizeof(double);
  fk::vector<double> values(num_values);

  infile.read(reinterpret_cast<char *>(values.data()), bytes);

  if constexpr (std::is_same_v<P, double>)
  {
    return values;
  }
  else
  {
    fk::vector<P> converted_values(values.size());
    for (int i = 0; i < converted_values.size(); i++)
      converted_values[i] = static_cast<P>(values[i]);
    return converted_values;
  }

  // infile implicitly closed on exit
}

//
// these ascii files can be generated in octave with, e.g.,
//
// w = 2
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
double read_scalar_from_txt_file(std::filesystem::path const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: scalar"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "scalar");

  double value;

  infile >> value;

  return value;
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
template<typename P>
fk::vector<P> read_vector_from_txt_file(std::filesystem::path const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // make sure we're working with a (either row or column) vector
  expect((rows == 1) || columns == 1);

  int const num_elems = (rows >= columns) ? rows : columns;

  // create output vector
  fk::vector<P> values(num_elems);

  double x = 0;
  for (auto i = 0; i < num_elems; ++i)
  {
    infile >> x;
    values(i) = static_cast<P>(x);
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
template<typename P>
fk::matrix<P> read_matrix_from_txt_file(std::filesystem::path const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // create output matrix
  fk::matrix<P> values(rows, columns);

  for (auto i = 0; i < rows; ++i)
    for (auto j = 0; j < columns; ++j)
    {
      infile >> tmp_str;
      values(i, j) = static_cast<P>(std::stod(tmp_str));
    }

  return values;
}

// stitch matrices together side by side (all must have same # rows)
template<typename P>
fk::matrix<P> horz_matrix_concat(std::vector<fk::matrix<P>> const &matrices)
{
  expect(matrices.size() > 0);
  auto const [nrows, ncols] = [&]() {
    int col_accum            = 0;
    int const matrices_nrows = matrices[0].nrows();
    for (auto const &mat : matrices)
    {
      col_accum += mat.ncols();
      expect(mat.nrows() == matrices_nrows);
    }
    return std::array<int, 2>{matrices_nrows, col_accum};
  }();
  fk::matrix<P> concat(nrows, ncols);
  int col_index = 0;
  for (auto const &mat : matrices)
  {
    concat.set_submatrix(0, col_index, mat);
    col_index += mat.ncols();
  }
  return concat;
}

template<typename P>
fk::vector<P> interp1(fk::vector<P> const &sample, fk::vector<P> const &values,
                      fk::vector<P> const &coords)
{
  // nearest neighbor 1D interpolation with extrapolation
  // equivalent to matlab's interp1(sample, values, coords, 'nearest', 'extrap')
  expect(sample.size() == values.size());
  expect(coords.size() >= 1);

  // output vector
  fk::vector<P> interpolated(coords.size());

  // perform interpolation at each query point
  for (int i = 0; i < coords.size(); ++i)
  {
    P const query = coords[i];

    // find minimum distance
    P min_distance = std::fabs(sample[0] - query);
    int min_ind    = 0;
    for (int j = 1; j < sample.size(); ++j)
    {
      // compute distance from query point to sample points, minimum value is
      // the nearest neighbor to use
      P const distance = std::fabs(sample[j] - query);
      // matlab seems to take the last points that are equal to the min, so <=
      // is used here
      if (distance <= min_distance)
      {
        min_distance = distance;
        min_ind      = j;
      }
    }

    interpolated(i) = values[min_ind];
  }

  return interpolated;
}

// explicit instantiations
#ifdef ASGARD_ENABLE_DOUBLE
template fk::vector<double>
read_vector_from_bin_file(std::filesystem::path const &path);
template fk::vector<double>
read_vector_from_txt_file(std::filesystem::path const &path);
template fk::matrix<double>
read_matrix_from_txt_file(std::filesystem::path const &path);

template fk::matrix<double>
horz_matrix_concat(std::vector<fk::matrix<double>> const &matrices);

template fk::vector<double> interp1(fk::vector<double> const &sample,
                                    fk::vector<double> const &values,
                                    fk::vector<double> const &coords);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template fk::vector<float>
read_vector_from_bin_file(std::filesystem::path const &);
template fk::vector<float>
read_vector_from_txt_file(std::filesystem::path const &);
template fk::matrix<float>
read_matrix_from_txt_file(std::filesystem::path const &);

template fk::vector<float> interp1(fk::vector<float> const &sample,
                                   fk::vector<float> const &values,
                                   fk::vector<float> const &coords);
#endif

template fk::vector<int>
read_vector_from_bin_file(std::filesystem::path const &);
template fk::vector<int>
read_vector_from_txt_file(std::filesystem::path const &);
template fk::matrix<int>
read_matrix_from_txt_file(std::filesystem::path const &);

template fk::matrix<int>
horz_matrix_concat(std::vector<fk::matrix<int>> const &matrices);
template fk::matrix<float>
horz_matrix_concat(std::vector<fk::matrix<float>> const &matrices);

} // namespace asgard
