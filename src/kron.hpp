#include "tensors.hpp"
#include "batch.hpp"
#include "fast_math.hpp"

#include <numeric>
#include <array>
#include <iostream>

template< typename P, mem_type mem, resource resrc >
int calculate_workspace_len( std::vector< fk::matrix< P, mem, resrc > > const &matrix,
                             int const x_size )
{
  int greatest = x_size;
  int r_prod = 1;
  int c_prod = 1;

  typename std::vector< fk::matrix< P, mem, resrc > >::const_reverse_iterator iter;

  for( iter = matrix.rbegin(); iter != matrix.rend(); ++iter )
  {
    c_prod *= iter->ncols();
    r_prod *= iter->nrows();

    int const size = x_size / c_prod * r_prod;
    if( size > greatest ) greatest = size;
  }

  return greatest;
}

template< typename P, mem_type mem >
fk::vector< P, mem_type::owner, resource::device >
kron( std::vector< fk::matrix< P, mem, resource::device > > const &matrix, 
      fk::vector< P, mem, resource::device > const &x )
{
  assert( matrix.size() > 0 );

  /* ensure "x" is correct size */
  int const x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, fk::matrix< P, mem, resource::device > const &m )
                           {
                             return i * m.ncols();
                           } );
  assert( x.size() == x_size );

  /* determine correct workspace length */
  int const workspace_len = calculate_workspace_len( matrix, x_size );

  /* Captain! Does this need to be on the device? */
  /* set withk a view */
  fk::vector<P, mem_type::owner, resource::device > w0(workspace_len);
  w0.set_subvector( 0, x );
  fk::vector<P, mem_type::owner, resource::device > w1(workspace_len);

  std::array< fk::vector< P, mem_type::owner, resource::device >, 2 > workspace = { w0, w1 };

  int in = 0;
  int out = 1;
  int stride = 1;
  int v_size = x_size;
  
  typename std::vector< fk::matrix< P, mem, resource::device > >::const_reverse_iterator iter;
  for( iter = matrix.rbegin(); iter != matrix.rend(); ++iter )
  {
    int const read_stride = stride * iter->ncols();
    int const write_stride = stride * iter->nrows();
    int const n_gemms = v_size / read_stride;

    /* create a batch of matrices */
    batch< P > left_side( n_gemms, stride, iter->ncols(), stride, false );
    batch< P > right_side( n_gemms, iter->nrows(), 
                                           iter->ncols(), iter->stride(), true );
    batch< P > product( n_gemms, stride, iter->nrows(), stride, false );

    for( int j = 0; j < n_gemms; ++j )
    {
      fk::matrix< P, mem, resource::device >
      input( workspace[ in ], stride, iter->ncols(), j * read_stride );

      fk::matrix< P, mem, resource::device >
      output( workspace[ out ], stride, iter->nrows(), j * write_stride );

      /* use an is_same statement here to figure out whether to use a view constructor or not */
      left_side.assign_entry(input, j);
      right_side.assign_entry( fk::matrix< P, mem_type::view, resource::device >( (*iter ) ), j );
      product.assign_entry( output, j );
    }
    
    /* new code */
    batched_gemm< P >( left_side, right_side, product, 1, 0 );

    v_size = n_gemms * write_stride;
    stride = write_stride;

    int const tmp = in;
    in = out;
    out = tmp;
  }

  int const y_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, fk::matrix< P, mem, resource::device > const &m )
                           {
                             return i * m.nrows();
                           } );
  
  return workspace[in].extract(0, y_size - 1);
}
