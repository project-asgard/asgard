#include "tests_general.hpp"
#include "kron.hpp"

TEMPLATE_TEST_CASE( "kron", "[kron]", double, float )
{
  SECTION("calculate_workspace_size")
  {
    fk::matrix< TestType > a( 5, 10 );
    fk::matrix< TestType > b( 5, 10 );
    fk::matrix< TestType > c( 5, 10 );
    fk::matrix< TestType > e( 10, 5 );
    fk::matrix< TestType > f( 10, 5 );

    std::vector< fk::matrix< TestType, mem_type::owner > > matrix =
    { a, b, c, e, f };

    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType > &m )
                                 {
                                   return i * m.ncols();
                                 } );
    int correct_size = 1e5;

    REQUIRE( calculate_workspace_len( matrix, x_size ) == correct_size );
  }

  /* a simple 2 matrix and vector kron */
  SECTION("kron_0")
  {
    fk::matrix< TestType, mem_type::owner, resource::device > a =
    { { 2, 3 }, { 4, 5 } };

    fk::matrix< TestType, mem_type::owner, resource::device > b =
    { { 6, 7 }, { 8, 9 } };

    const std::vector< fk::matrix< TestType, mem_type::view, resource::device > > matrix =
    { fk::matrix<TestType, mem_type::view, resource::device >(a),
      fk::matrix<TestType, mem_type::view, resource::device >(b) };

    fk::vector< TestType, mem_type::owner, resource::device > x = { 10, 11, 12, 13 };

    fk::vector< TestType, mem_type::owner, resource::host > correct = { 763, 997, 1363, 1781 };

    fk::vector< TestType, mem_type::owner, resource::device > r = 
    kron< TestType, mem_type::view >( matrix, 
                                      fk::vector<TestType, mem_type::view, resource::device >(x) );

    REQUIRE( r.clone_onto_host() == correct );
  }

  SECTION("kron_1")
  {
    auto matrix_all_twos =
    []( int rows, int cols )-> fk::matrix< TestType, mem_type::owner, resource::device >
    {
      fk::matrix< TestType, mem_type::owner, resource::host> m( rows, cols );

      for( int i = 0; i < rows; ++i )
      {
        for( int j = 0; j < cols; ++j )
        {
          m( i, j ) = 2;
        }
      }

      return m.clone_onto_device();
    };

    auto m0 = matrix_all_twos( 3, 4 );
    auto m1 = matrix_all_twos( 5, 3 );
    auto m2 = matrix_all_twos( 2, 7 );
    auto m3 = matrix_all_twos( 9, 6 );
    auto m4 = matrix_all_twos( 12, 3 );
    auto m5 = matrix_all_twos( 4, 14 );
    auto m6 = matrix_all_twos( 10, 3 );
    auto m7 = matrix_all_twos( 6, 5 );
    fk::matrix<TestType, mem_type::owner, resource::device > m8 = { {3, 3}, {3, 3} };

    std::vector< fk::matrix< TestType, mem_type::view, resource::device > > matrix =
    { 
      fk::matrix<TestType, mem_type::view, resource::device >( m0 ), 
      fk::matrix<TestType, mem_type::view, resource::device >( m1 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m2 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m3 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m4 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m5 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m6 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m7 ),
      fk::matrix<TestType, mem_type::view, resource::device >( m8 )
    };

    /* Captain! see if you can use auto to fix the compiler warning 
       that will be generated below */
    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType, mem_type::view > &m )
                                 {
                                   return i * m.ncols();
                                 } );

    int y_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType, mem_type::view > &m )
                                 {
                                   return i * m.nrows();
                                 } );

    /* could you make this a view instead of an owner? */
    fk::vector< TestType, mem_type::owner, resource::host >
    x( std::vector< TestType >( x_size, 1 ) );

    auto r =
    kron< TestType, mem_type::view >( matrix, 
                                      fk::vector< TestType, 
                                                  mem_type::view,
                                                  resource::device >(x.clone_onto_device() ) );

    fk::vector< TestType > 
    correct( std::vector< TestType >( y_size, x_size * ( 1 << ( matrix.size() - 1 ) ) * 3 ) );

    REQUIRE( r.clone_onto_host() == correct );
  }
}
