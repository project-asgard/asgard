/* Captain! Make a detailed algorithm paper describing how this works */
/*
1. Write detailed algorithm paper
2. Test edge cases
3. Make programmatic test with large matrix of all ones or many zeroes
4. Make a better transpose
5. Replace recursive kron with this one and make sure all tests pass
6. Remove recursive kron entirely
7. Move this kron to transformations component? Or batching? Set it up to easily be
   batchable.
8. Incorporate batching
9. Address uneccesary copies
*/
#include "kron.hpp"
#include <numeric>
#include <array>

template< typename P >
int calculate_workspace_len( std::vector< fk::matrix< P, mem_type::owner > > const &matrix,
                             int const x_size )
{
  int greatest = x_size;
  int r_prod = 1;
  int c_prod = 1;

  typename std::vector< fk::matrix< P, mem_type::owner > >::const_reverse_iterator iter;

  for( iter = matrix.rbegin(); iter != matrix.rend(); ++iter )
  {
    c_prod *= iter->ncols();
    r_prod *= iter->nrows();

    int const size = x_size / c_prod * r_prod;
    if( size > greatest ) greatest = size;
  }

  return greatest;
}

template< typename P >
fk::vector< P, mem_type::owner >
kron( std::vector< fk::matrix< P, mem_type::owner > > const &matrix, 
      fk::vector< P, mem_type::owner > const &x )
{
  /* check "x" is correct size */
  int const x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, fk::matrix< P > const &m )
                           {
                             return i * m.ncols();
                           } );

  assert( x.size() == x_size );

  /* determine correct workspace length */
  int const workspace_len = calculate_workspace_len( matrix, x_size );
  //std::cout << "workspace_len: "  << workspace_len << std::endl;
  fk::vector<P> w0(workspace_len);
  /* Captain! is a copy of x made in set_subvector api? */
  w0.set_subvector( 0, x );
  fk::vector<P> w1(workspace_len);

  std::array< fk::vector< P >, 2 > workspace = { w0, w1 };

  int in = 0;
  int out = 1;
  int stride = 1;
  /* size of input vector at each stage */
  int v_size = x_size;
  
  typename std::vector< fk::matrix< P, mem_type::owner > >::const_reverse_iterator iter;
  for( iter = matrix.rbegin(); iter != matrix.rend(); ++iter )
  {
    int const read_stride = stride * iter->ncols();
    int const write_stride = stride * iter->nrows();
    int const n_gemms = v_size / read_stride;

    /* Captain! Test code */
    /*
    std::cout << "iteration " 
              << "\nmatrix: ( " << iter->nrows() << ", " << iter->ncols() << " )"
              << "\nv_size: " << v_size 
              << "\nstride: " << stride 
              << "\nread_stride: " << read_stride 
              << "\nwrite_stride: " << write_stride
              << "\nn_gemms: " << n_gemms 
              << "\n" << std::endl;

    std::cout << "matrix:" << std::endl;
    iter->print();
    */
    /* end test code */

    for( int j = 0; j < n_gemms; ++j )
    {
      fk::matrix< P, mem_type::view >
      input( workspace[ in ], stride, iter->ncols(), j * read_stride );

      fk::matrix< P, mem_type::view >
      output( workspace[ out ], stride, iter->nrows(), j * write_stride );

      fk::matrix< P, mem_type::owner > output_copy( output.ncols(), output.nrows() );

      fm::gemm< P, mem_type::owner, mem_type::view, mem_type::owner >
      ( (*iter), input, output_copy, false, true );

      output_copy.transpose();
      output = output_copy;

      /* Captain! Test code */
      /*
      std::cout << "input( " << in << " ):" << std::endl;
      input.print();
      std::cout << "output( " << out << " ):" << std::endl;
      output.print();
      */
      /* end test code */
    }

    v_size = n_gemms * write_stride;
    stride = write_stride;

    int const tmp = in;
    in = out;
    out = tmp;
  }

  int const y_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                           []( int const i, fk::matrix< P > const &m )
                           {
                             return i * m.nrows();
                           } );
  
  /* Captain! Is returning "in" correct for both even and odd numbers of matrices? */
  return workspace[in].extract(0, y_size - 1);
}

template
fk::vector< double, mem_type::owner >
kron( std::vector< fk::matrix< double, mem_type::owner > > const &matrix, 
      fk::vector< double, mem_type::owner > const &x );

template
fk::vector< float, mem_type::owner >
kron( std::vector< fk::matrix< float, mem_type::owner > > const &matrix, 
      fk::vector< float, mem_type::owner > const &x );

template
int calculate_workspace_len( std::vector< fk::matrix< float, mem_type::owner > > 
                             const &matrix, int const x_size );

template
int calculate_workspace_len( std::vector< fk::matrix< double, mem_type::owner > > 
                             const &matrix, int const x_size );
