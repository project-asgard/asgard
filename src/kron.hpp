#include "tensors.hpp"
#include "batch.hpp"
#include "fast_math.hpp"

#include <numeric>
#include <array>
#include <iostream>

template< typename P, resource resrc >
int calculate_workspace_len( std::vector< fk::matrix< P, mem_type::view, resrc > > const &matrix,
                             int const x_size )
{
  int greatest = x_size;
  int r_prod = 1;
  int c_prod = 1;

  typename std::vector< fk::matrix< P, mem_type::view, resrc > >::const_reverse_iterator iter;

  for( iter = matrix.rbegin(); iter != matrix.rend(); ++iter )
  {
    c_prod *= iter->ncols();
    r_prod *= iter->nrows();

    int const size = x_size / c_prod * r_prod;
    if( size > greatest ) greatest = size;
  }

  return greatest;
}

/* convenience storage class */
template< typename P >
class batch_set
{
  public:

    batch_set( batch< P > &&left, batch< P > &&right, batch< P > &&product )
      :
      left( left ), right( right ), product( product )
    {}

    batch_set( batch<P> &&bs )
      :
      left( std::move( bs.left ) ),
      right( std::move( bs.right ) ),
      product( std::move( bs.product ) )
    {
    }

    batch< P > left;
    batch< P > right;
    batch< P > product;
};

template< typename P, resource resrc >
class batch_job
{
  public:

    batch_job( P const alpha,
               P const beta,
               int const workspace_len,
               fk::vector< P, mem_type::view, resrc > const &x,
               int const y_size )
      :
      alpha( alpha ),
      beta( beta ),
      in( 0 ),
      out( 1 ),
      y_size( y_size ),
      workspace( { fk::vector< P, mem_type::owner, resrc >( workspace_len ), 
                   fk::vector< P, mem_type::owner, resrc >( workspace_len ) } )
    {
      fk::vector< P, mem_type::view, resrc > init_w0( workspace[ 0 ], 0, x.size() - 1 );
      init_w0 = x;
    }

    void add_batch_set( batch_set< P > &&bs )
    {
      batches.emplace_back( bs );

      int const tmp = in;
      in = out;
      out = tmp;

      return;
    }

    fk::vector< P, mem_type::owner, resrc > const &get_input_workspace()
    {
      return workspace[ in ];
    } 

    fk::vector< P, mem_type::owner, resrc > const &get_output_workspace()
    {
      return workspace[ out ];
    } 

    std::vector< batch_set< P > > batches;
    P alpha;
    P beta;
    int in;
    int out;
    int y_size;
    std::array< fk::vector< P, mem_type::owner, resrc >, 2 > workspace;
};

template< typename P, resource resrc >
fk::vector< P, mem_type::owner, resrc >
execute_batch_job( batch_job< P, resrc > &bj )
{
  for( auto const &bs : bj.batches )
  {
    batched_gemm( bs.left, bs.right, bs.product, bj.alpha, bj.beta );
  }

  fk::vector< P, mem_type::view, resrc > v( bj.get_input_workspace(), 0, bj.y_size - 1 );

  fk::vector< P, mem_type::owner, resrc > r(v);

  return r;
}

template< typename P, resource resrc >
batch_job< P, resrc >
kron_batch( std::vector< fk::matrix< P, mem_type::view, resrc > > const &matrix, 
      fk::vector< P, mem_type::view, resrc > const &x )
{
  assert( matrix.size() > 0 );

  /* ensure "x" is correct size */
  assert( x.size() == std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, auto const &m )
                           {
                             return i * m.ncols();
                           } ) );

  /* determine correct workspace length */
  int const workspace_len = calculate_workspace_len( matrix, x.size() );

  int const y_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, auto const &m )
                           {
                             return i * m.nrows();
                           } );

  batch_job< P, resrc > job( 1, 0, workspace_len, x, y_size );

  /* Below is a loop unrolled one iteration */
  typename 
  std::vector< fk::matrix< P, mem_type::view, resource::device > >::const_reverse_iterator iter =
  matrix.rbegin();

  {
    int const rows = x.size() / iter->ncols();

    batch< P > left( 1, iter->nrows(), iter->ncols(), iter->stride(), false );
    batch< P > right( 1, iter->ncols(), rows, iter->ncols(), false );
    batch< P > product( 1, iter->nrows(), rows, iter->nrows(), false );

    fk::matrix< P, mem_type::view, resource::device >
    input( job.get_input_workspace(), iter->ncols(), rows, 0 );

    fk::matrix< P, mem_type::view, resource::device >
    output( job.get_output_workspace(), iter->nrows(), rows, 0 );

    right.assign_entry(input, 0);
    left.assign_entry( fk::matrix< P, mem_type::view, resource::device >( (*iter ) ), 0);
    product.assign_entry( output, 0 );

    batch_set< P > bs( std::move( left ), std::move( right ), std::move( product ) );

    job.add_batch_set( std::move( bs ) );
  }

  int stride = iter->nrows();
  int v_size = x.size() / iter->ncols() * iter->nrows();
  ++iter;

  for( ; iter != matrix.rend(); ++iter )
  {
    int const read_stride = stride * iter->ncols();
    int const write_stride = stride * iter->nrows();
    int const n_gemms = v_size / read_stride;

    /* create a batch of matrices */
    batch< P > left( n_gemms, stride, iter->ncols(), stride, false );
    batch< P > right( n_gemms,
                      iter->nrows(), 
                      iter->ncols(), iter->stride(), true );
    batch< P > product( n_gemms, stride, iter->nrows(), stride, false );

    for( int j = 0; j < n_gemms; ++j )
    {
      fk::matrix< P, mem_type::view, resource::device >
      input( job.get_input_workspace(), stride, iter->ncols(), j * read_stride );

      fk::matrix< P, mem_type::view, resource::device >
      output( job.get_output_workspace(), stride, iter->nrows(), j * write_stride );

      left.assign_entry(input, j);
      right.assign_entry( fk::matrix< P, mem_type::view, resource::device >( (*iter ) ), j);
      product.assign_entry( output, j );
    }

    batch_set< P > bs( std::move( left ), std::move( right ), std::move( product ) );

    job.add_batch_set( std::move( bs ) );
    
    v_size = n_gemms * write_stride;
    stride = write_stride;
  }

  return job;
}
