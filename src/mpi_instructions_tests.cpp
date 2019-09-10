#include "mpi_instructions.hpp"
#include "tests_general.hpp"

#include <set>

/* ensures that each segment described in c_stop is encoded via possibly
   multiple intervals
   in r_stop. Ensures that the r_stop continuously represents the entire range
 */
bool check_row_space_intervals(
    const std::vector<std::vector<class mpi_node_and_range>>
        &row_space_intervals,
    const std::vector<int> &c_stop, const std::vector<int> &r_stop)
{
  int r_start = 0;

  int c_start = 0;

  int c_end;

  for (int i = 0; i < (int)row_space_intervals.size(); i++)
  {
    c_end = c_stop[i];

    /* first iteration unrolled */
    const class mpi_node_and_range &nar = row_space_intervals[i][0];

    int prev_start = nar.start + r_start;

    int initial_start = prev_start;

    if (initial_start != c_start)
      return false;

    int prev_stop = nar.stop + r_start;

    /* remaining iterations */
    for (int j = 1; j < (int)row_space_intervals[i].size(); j++)
    {
      const class mpi_node_and_range &nar = row_space_intervals[i][j];

      r_start = r_stop[nar.linear_index - 1] + 1;

      prev_start = nar.start + r_start;

      if (prev_start != prev_stop + 1)
        return false;

      prev_stop = nar.stop + r_start;
    }

    if (prev_stop != c_end)
      return false;

    /* for next loop iteration */
    c_start = c_end + 1;
  }

  return true;
}

/* return a slice that corresponds to the range specified in nar */
std::vector<int>
cut_a_slice( mpi_node_and_range const &nar, const std::vector<int> &x )
{
  std::vector<int> slice;

  std::vector<int>::const_iterator iter_0 = x.begin();

  std::vector<int>::const_iterator iter_1 = x.begin();

  std::advance(iter_0, nar.start);

  /* +1 because copy range is [ first iterator, last iterator ) */
  std::advance(iter_1, nar.stop + 1);

  std::copy(iter_0, iter_1, std::back_inserter(slice));

  return slice;
}

/* returns correct vector slice for column */
std::vector<int>
correct_slice(int c, const std::vector<int> &c_stop, const std::vector<int> &x)
{
  std::vector<int> slice;

  /* the first slice from x has been requested */
  if (c == 0)
  {
    std::vector<int>::const_iterator iter = x.begin();

    std::advance(iter, c_stop[0] + 1);

    std::copy(x.begin(), iter, std::back_inserter(slice));

    return slice;
  }

  else
  {
    std::vector<int>::const_iterator iter_0 = x.begin();

    std::vector<int>::const_iterator iter_1 = x.begin();

    std::advance(iter_0, c_stop[c - 1] + 1);

    std::advance(iter_1, c_stop[c] + 1);

    std::copy(iter_0, iter_1, std::back_inserter(slice));

    return slice;
  }
}

/* ensure that each process node can construct its slice of the vector based
   only on what it receives */
bool check_slices(const std::vector< std::vector< mpi_message > > &mpi_instructions,
                  std::vector< int > const &r_stop,
                  std::vector< int > const &c_stop )
{
  /* create fake data */
  std::vector<int> x;

  for (int i = 0; i < c_stop.back(); i++) x.push_back(i);

  for (int r = 0; r < (int)r_stop.size(); r++)
  {
    for (int c = 0; c < (int)c_stop.size(); c++)
    {
      std::vector< mpi_message > const &mpi_messages =
      mpi_instructions[ r * c_stop.size() + c ];

      std::vector<int> derived_slice;

      /* only the receive-type mpi_messages are of interest here */
      for (int i = 0; i < (int)mpi_messages.size(); i++)
      {
        if (mpi_messages[i].mpi_message_type == mpi_message_enum::receive)
        {
          std::vector<int> sub_slice =
              cut_a_slice(mpi_messages[ i ].nar, x);

          std::copy(sub_slice.begin(), sub_slice.end(),
                    std::back_inserter(derived_slice));
        }
      }

      std::vector<int> goal_slice = correct_slice(c, c_stop, x);

      if( derived_slice.size() != goal_slice.size() ) return false;

      else if (derived_slice != goal_slice) return false;
    }
  }

  return true;
}

/* ensure that every send has a matching receive with the same info */
bool check_mpi_messages(const std::vector< std::vector< mpi_message > > &mpi_messages,
                  std::vector< int > const &r_stop,
                  std::vector< int > const &c_stop )
{
  /* these arrays will have the format:
     { sending linear index, receiving linear index, start, stop } */
  std::set<std::array<const int, 4>> receive_mpi_message;

  std::vector<std::array<const int, 4>> send_mpi_message;

  for (int r = 0; r < (int)r_stop.size(); r++)
  {
    for (int c = 0; c < (int)c_stop.size(); c++)
    {
      const std::vector<class mpi_message> &mpi_message =
      mpi_messages[ r * c_stop.size() + c ];

      for (int i = 0; i < (int)mpi_message.size(); i++)
      {
        const class mpi_node_and_range &nar = mpi_message[i].nar;

        if (mpi_message[i].mpi_message_type == mpi_message_enum::send)
        {
          const std::array<const int, 4> array{
              r * (int)c_stop.size() + c, nar.linear_index,
              nar.start, nar.stop};

          send_mpi_message.emplace_back(std::move(array));
        }

        else if (mpi_message[i].mpi_message_type == mpi_message_enum::receive)
        {
          std::array<const int, 4> array{nar.linear_index,
                                         r * (int)c_stop.size() + c,
                                         nar.start, nar.stop};

          receive_mpi_message.emplace(std::move(array));
        }
      }
    }
  }

  if (send_mpi_message.size() != receive_mpi_message.size())
    return false;

  for (int i = 0; i < (int)send_mpi_message.size(); i++)
  {
    if (receive_mpi_message.find(send_mpi_message[i]) ==
        receive_mpi_message.end())
      return false;
  }

  return true;
}

bool check_correct_intervals( std::vector< std::vector< mpi_node_and_range > > const &check,
                              std::vector< std::vector< mpi_node_and_range > > const &correct )
{
  if( check.size() != correct.size() ) return false; 

  else for( int i = 0; i < (int)check.size(); i++ )
  {
    if( check[ i ].size() != correct[ i ].size() ) return false;

    else for( int j = 0; j < (int)check[ i ].size(); j++ )
    {
      mpi_node_and_range const &check_nar = check[ i ][ j ];

      mpi_node_and_range const &correct_nar = correct[ i ][ j ];

      if( check_nar.linear_index != correct_nar.linear_index ) return false;
      if( check_nar.start != correct_nar.start ) return false;
      if( check_nar.stop != correct_nar.stop ) return false;
    }
  }

  return true;
}

TEST_CASE("mpi_mpi_messages", "[mpi]")
{
  /* test data - 3 sets of r_stop, c_stop, and correct_intervals */
  std::vector< std::vector< int > > c_stops =
  {{ 12, 18, 20, 36 }, { 5, 10, 15 }, { 10, 23, 47, 100 }};

  std::vector< std::vector< int > > r_stops =
  {{ 4, 8, 12, 20, 32, 36 }, { 7, 15 }, { 25, 50, 90, 100 }};

  std::vector< std::vector< std::vector< mpi_node_and_range > > > correct_intervals_v =
  {
  {{ mpi_node_and_range( 0, 0, 4 ),
     mpi_node_and_range( 1, 5, 8 ),
     mpi_node_and_range( 2, 9, 12 ) },
   { mpi_node_and_range( 3, 13, 18 ) },
   { mpi_node_and_range( 3, 19, 20 ) },
   { mpi_node_and_range( 4, 21, 32 ),
     mpi_node_and_range( 5, 33, 36 ) }},
  {{ mpi_node_and_range( 0, 0, 5 ) },
   { mpi_node_and_range( 0, 6, 7 ), mpi_node_and_range( 1, 8, 10 ) },
   { mpi_node_and_range( 1, 11, 15 ) }},
  {{ mpi_node_and_range( 0, 0, 10 ) },
   { mpi_node_and_range( 0, 11, 23 ) },
   { mpi_node_and_range( 0, 24, 25 ), mpi_node_and_range( 1, 26, 47 ) }, 
   { mpi_node_and_range( 1, 48, 50 ), mpi_node_and_range( 2, 51, 90 ), 
     mpi_node_and_range( 3, 91, 100 ) }}
  };

  std::vector< std::vector< mpi_node_and_range > > correct_intervals =
  {{ mpi_node_and_range( 0, 0, 4 ),
     mpi_node_and_range( 1, 5, 8 ),
     mpi_node_and_range( 2, 9, 12 ) },
   { mpi_node_and_range( 3, 13, 18 ) },
   { mpi_node_and_range( 3, 19, 20 ) },
   { mpi_node_and_range( 4, 21, 32 ),
     mpi_node_and_range( 5, 33, 36 ) }};

  SECTION("rowspace_intervals")
  {
    bool pass = true;

    for( int i = 0; i < (int)r_stops.size(); i++ )
    {
      const std::vector<std::vector<class mpi_node_and_range>>
      row_space_intervals = generate_row_space_intervals( r_stops[ i ], c_stops[ i ] );

      pass = pass && check_correct_intervals( row_space_intervals, correct_intervals_v[ i ] );
    }

    REQUIRE(pass == true);
  }

  SECTION("build_slices")
  {
    bool pass = true;

    for( int i = 0; i < (int)r_stops.size(); i++ )
    {
      std::vector< std::vector< mpi_message > > const mpi_messages =
      generate_mpi_messages( r_stops[ i ], c_stops[ i ] );

      pass = pass && check_slices( mpi_messages, r_stops[ i ], c_stops[ i ] );
    }

    REQUIRE(pass == true);
  }

  SECTION("check_mpi_messages")
  {
    bool pass = true;

    for( int i = 0; i < (int)r_stops.size(); i++ )
    {
      std::vector< std::vector< mpi_message > > const mpi_messages =
      generate_mpi_messages( r_stops[ i ], c_stops[ i ] );

      pass = pass && check_mpi_messages( mpi_messages, r_stops[ i ], c_stops[ i ] );
    }

    REQUIRE(pass == true);
  }
}
