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

  /* Captain! Run a test with an empty kron list and other edge cases */

  /* a simple 2 matrix and vector kron */
  SECTION("kron_0")
  {
    fk::matrix< TestType > a =
    { { 2, 3 }, { 4, 5 } };

    fk::matrix< TestType > b =
    { { 6, 7 }, { 8, 9 } };

    std::vector< fk::matrix< TestType > > matrix = { a, b };

    fk::vector< TestType > x = { 10, 11, 12, 13 };

    fk::vector< TestType > correct = { 763, 997, 1363, 1781 };

    fk::vector< TestType > r = kron( matrix, x );

    REQUIRE( r == correct );
  }

  /* Just to check correct problem parameters */
  SECTION("kron_1")
  {
    fk::matrix< TestType > a( 3, 8 );
    fk::matrix< TestType > b( 5, 10 );
    fk::matrix< TestType > c( 7, 4 );
    fk::matrix< TestType > e( 9, 6 );
    fk::matrix< TestType > f( 11, 12 );

    std::vector< fk::matrix< TestType, mem_type::owner > > matrix =
    { a, b, c, e, f };

    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType > &m )
                                 {
                                   return i * m.ncols();
                                 } );

    fk::vector< TestType > x( x_size );

    /* Captain! Add a check for correct size */
    //fk::vector< TestType > r = kron( matrix, x );
  }

  /* hacky ass way to test recursive kron correctness with existing routine */
  SECTION("kron_2")
  {
    /* 2, 3 */
    fk::matrix< TestType > a =
    {
    { 1, 1 }, { 1, 1 }
    };
    /* 5, 4 */
    fk::matrix< TestType > b =
    {
    { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, {1, 1, 1, 1 }, { 1, 1, 1, 1}, { 1, 1, 1, 1 }
    };
    /* 3, 7 */
    fk::matrix< TestType > c =
    {
    { 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1, 1, 1 }
    };
    /* 10, 2 */
    fk::matrix< TestType > e =
    {
    { 1, 1 }, {1, 1}, { 1, 1 }, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, { 1, 1 }
    };
    /* 2, 6 */
    fk::matrix< TestType > f =
    {
    {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}
    };

    std::vector< fk::matrix< TestType, mem_type::owner > > matrix =
    { a, b, c, e, f };

    std::vector< fk::matrix< TestType, mem_type::view > > matrix_views =
    { fk::matrix< TestType, mem_type::view >( a ),
      fk::matrix< TestType, mem_type::view >( b ),
      fk::matrix< TestType, mem_type::view >( c ),
      fk::matrix< TestType, mem_type::view >( e ),
      fk::matrix< TestType, mem_type::view >( f ) };

    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType > &m )
                                 {
                                   return i * m.ncols();
                                 } );

    std::vector< TestType > x_vec(x_size);
    std::iota( x_vec.begin(), x_vec.end(), 1 );
    fk::vector< TestType > x( x_vec );

    fk::vector< TestType > r = kron( matrix, x );
    fk::matrix< TestType > m = recursive_kron( matrix_views );
    fk::vector< TestType > r_correct = m * x;
    REQUIRE( r == r_correct );
  }

  SECTION("kron_3")
  {
    /* 2, 3 */
    fk::matrix< TestType > a =
    {
    { 1, 2 }, { 3, 4 }
    };
    /* 5, 4 */
    fk::matrix< TestType > b =
    {
    { 3, 6, 3, 6 }, { 9, 4, 7, 4 }, {6, 3, 4, 9 }, { 1, 3, 7, 11}, { 5, 6, 5, 1 }
    };
    /* 3, 7 */
    fk::matrix< TestType > c =
    {
    { 1, 2, 3, 4, 5, 6, 7 }, { 7, 6, 4, 5, 2, 3, 1 }, { 8, 9, 7, 3, 5, 1, 8 }
    };
    /* 10, 2 */
    fk::matrix< TestType > e =
    {
    { 7, 8 }, {7, 9}, { 8, 7 }, {3, 1}, {2, 5}, {8, 8}, {4, 6}, {7, 4}, {4, 5}, { 2, 3 }
    };

    std::vector< fk::matrix< TestType, mem_type::owner > > matrix =
    { a, b, c, e };

    std::vector< fk::matrix< TestType, mem_type::view > > matrix_views =
    { fk::matrix< TestType, mem_type::view >( a ),
      fk::matrix< TestType, mem_type::view >( b ),
      fk::matrix< TestType, mem_type::view >( c ),
      fk::matrix< TestType, mem_type::view >( e ) };

    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType > &m )
                                 {
                                   return i * m.ncols();
                                 } );

    std::vector< TestType > x_vec(x_size);
    std::iota( x_vec.begin(), x_vec.end(), 1 );
    fk::vector< TestType > x( x_vec );

    fk::vector< TestType > r = kron( matrix, x );
    fk::matrix< TestType > m = recursive_kron( matrix_views );
    fk::vector< TestType > r_correct = m * x;
    REQUIRE( r == r_correct );
  }

  SECTION("kron_4")
  {
    fk::matrix< TestType > a =
    {
    { 4, 3 }, { 2, 5 }
    };
    fk::matrix< TestType > b =
    {
    { 6, 7 }, { 8, 9 }
    };
    fk::matrix< TestType > c =
    {
    { 2, 3 }, { 4, 5 }
    };

    std::vector< fk::matrix< TestType, mem_type::owner > > matrix =
    { a, b, c };

    std::vector< fk::matrix< TestType, mem_type::view > > matrix_views =
    { fk::matrix< TestType, mem_type::view >( a ),
      fk::matrix< TestType, mem_type::view >( b ),
      fk::matrix< TestType, mem_type::view >( c ) };

    int x_size = std::accumulate( matrix.begin(), matrix.end(), 1, 
                                 []( int i, fk::matrix< TestType > &m )
                                 {
                                   return i * m.ncols();
                                 } );

    std::vector< TestType > x_vec(x_size);
    std::iota( x_vec.begin(), x_vec.end(), 1 );
    fk::vector< TestType > x( x_vec );

    fk::vector< TestType > r = kron( matrix, x );
    fk::matrix< TestType > m = recursive_kron( matrix_views );
    fk::vector< TestType > r_correct = m * x;

    r_correct.print();
    r.print();
    REQUIRE( r == r_correct );
  }
}
