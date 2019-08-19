#include "transformations.hpp"

#include "connectivity.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <numeric>

//
// set the range specified by first and last,
// starting with the supplied value and incrementing
// that value by stride for each position in the range.
//
template<typename ForwardIterator, typename P>
static void strided_iota(ForwardIterator first, ForwardIterator last, P value,
                         P const stride)
{
  while (first != last)
  {
    *first++ = value;
    value += stride;
  }
}

// perform recursive kronecker product
template<typename P>
fk::vector<P>
kron_d(std::vector<fk::vector<P>> const &operands, int const num_prods)
{
  assert(num_prods > 0);
  if (num_prods == 1)
  {
    return operands[0];
  }
  if (num_prods == 2)
  {
    return operands[0].single_column_kron(operands[1]);
  }
  return kron_d(operands, num_prods - 1)
      .single_column_kron(operands[num_prods - 1]);
}

// FIXME this function will need to change once dimensions can have different
// degree...
template<typename P>
fk::vector<P> combine_dimensions(int const degree, element_table const &table,
                                 std::vector<fk::vector<P>> const &vectors,
                                 P const time_scale)
{
  int const num_dims = vectors.size();
  assert(num_dims > 0);

  int64_t const vector_size =
      table.size() * static_cast<int64_t>(std::pow(degree, num_dims));
  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  assert(vector_size < INT_MAX);
  fk::vector<P> combined(vector_size);

  for (int i = 0; i < table.size(); ++i)
  {
    std::vector<fk::vector<P>> kron_list;
    fk::vector<int> coords = table.get_coords(i);
    for (int j = 0; j < num_dims; ++j)
    {
      // iterating over cell coords;
      // first num_dims entries in coords are level coords
      int const id          = get_1d_index(coords(j), coords(j + num_dims));
      int const index_start = id * degree;
      int const index_end   = ((id + 1) * degree) - 1;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    fk::vector<P> const partial_result =
        kron_d(kron_list, kron_list.size()) * time_scale;
    combined.set_subvector(i * std::pow(degree, num_dims), partial_result);
  }
  return combined;
}

// same as above, but only create the portion of the vector associated
// with the provided subgrid
template<typename P>
fk::vector<P> combine_dimensions(int const degree, element_table const &table,
                                 element_subgrid const &grid,
                                 std::vector<fk::vector<P>> const &vectors,
                                 P const time_scale)
{
  int const num_dims = vectors.size();
  assert(num_dims > 0);

  int64_t const vector_size = grid.nrows() * std::pow(degree, num_dims);

  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  assert(vector_size < INT_MAX);
  fk::vector<P> combined(vector_size);

  for (int i = grid.row_start; i <= grid.row_stop; ++i)
  {
    std::vector<fk::vector<P>> kron_list;
    fk::vector<int> coords = table.get_coords(i);
    for (int j = 0; j < num_dims; ++j)
    {
      // iterating over cell coords;
      // first num_dims entries in coords are level coords
      int const id          = get_1d_index(coords(j), coords(j + num_dims));
      int const index_start = id * degree;
      int const index_end   = ((id + 1) * degree) - 1;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    fk::vector<P> const partial_result =
        kron_d(kron_list, kron_list.size()) * time_scale;
    combined.set_subvector(grid.to_local_row(i) * std::pow(degree, num_dims),
                           partial_result);
  }
  return combined;
}

template fk::vector<double>
combine_dimensions(int const, element_table const &,
                   std::vector<fk::vector<double>> const &, double const);
template fk::vector<float>
combine_dimensions(int const, element_table const &,
                   std::vector<fk::vector<float>> const &, float const);

template fk::vector<double>
combine_dimensions(int const, element_table const &, element_subgrid const &,
                   std::vector<fk::vector<double>> const &, double const = 1.0);
template fk::vector<float>
combine_dimensions(int const, element_table const &, element_subgrid const &,
                   std::vector<fk::vector<float>> const &, float const = 1.0);
