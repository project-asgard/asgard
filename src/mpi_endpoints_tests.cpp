#include "mpi_endpoints.hpp"
#include "tests_general.hpp"

#include <set>

/* specifies num_segments intervals of interval_max length */
std::vector<int>
segment_range(int interval_max, int num_segments, std::random_device &rdev)
{
  std::vector<int> segment;

  int start = 0;

  while (num_segments > 0)
  {
    int n = (rdev() % interval_max) + 1;

    segment.push_back(start + n);

    start += n;

    num_segments--;
  }

  return segment;
}

std::vector<int> reverse_offset(std::vector<int> &c_stop)
{
  std::vector<int> r_stop;

  int last_index = c_stop.size() - 1;

  int start = 0;

  for (int i = 0; i < (int)c_stop.size() - 1; i++)
  {
    int j = last_index - i;

    int diff = c_stop[j] - c_stop[j - 1];

    r_stop.push_back(start + diff);

    start += diff;
  }

  r_stop.push_back(c_stop.back());

  return r_stop;
}

/* ensures that each segment described in c_stop is encoded via possibly
   multiple intervals
   in r_stop. Ensures that the r_stop continuously represents the entire range
 */
// clang-format off
bool check_row_space_intervals( const std::vector< std::vector< class node_and_range > >
                                &row_space_intervals,
                                const std::vector< int > &c_stop,
                                const std::vector< int > &r_stop )
// clang-format on
{
  int r_start = 0;

  int c_start = 0;

  int c_end;

  for (int i = 0; i < (int)row_space_intervals.size(); i++)
  {
    c_end = c_stop[i];

    /* first iteration unrolled */
    const class node_and_range &nar = row_space_intervals[i][0];

    int prev_start = nar.start + r_start;

    int initial_start = prev_start;

    if (initial_start != c_start)
      return false;

    int prev_stop = nar.stop + r_start;

    /* remaining iterations */
    for (int j = 1; j < (int)row_space_intervals[i].size(); j++)
    {
      const class node_and_range &nar = row_space_intervals[i][j];

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
// clang-format off
std::vector< int > cut_a_slice( const class node_and_range &nar,
                                const std::vector< int > &c_stop,
                                const std::vector< int > &r_stop,
                                const std::vector< int > &x )
// clang-format on
{
  std::vector<int> slice;

  /* convert nar's start and stop into global indices */
  /* get the column index */
  int r = nar.linear_index / c_stop.size();

  int first;

  if (r == 0)
  {
    first = 0;
  }

  else
    first = r_stop[r - 1] + 1;

  /* convert to global coordinates */
  int start = first + nar.start;

  int stop = first + nar.stop;

  /* copy those indices into slice and return it */
  std::vector<int>::const_iterator iter_0 = x.begin();

  std::vector<int>::const_iterator iter_1 = x.begin();

  std::advance(iter_0, start);

  /* +1 because copy range is [ first iterator, last iterator ) */
  std::advance(iter_1, stop + 1);

  std::copy(iter_0, iter_1, std::back_inserter(slice));

  return slice;
}

/* returns correct vector slice for column */
// clang-format off
std::vector< int > correct_slice( int c,
                                  const std::vector< int > &c_stop,
                                  const std::vector< int > &x )
// clang-format on
{
  /* Captain! Refactor to move code out of the if-else blocks */
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
bool check_slices(const class mpi_node_endpoints &mpi_node_endpoints)
{
  /* create fake data */
  std::vector<int> x;

  std::random_device rdev;

  const std::vector<int> &r_stop = mpi_node_endpoints.get_r_stop();

  const std::vector<int> &c_stop = mpi_node_endpoints.get_c_stop();

  for (int i = 0; i < c_stop.back(); i++)
    x.push_back(i);

  for (int r = 0; r < (int)r_stop.size(); r++)
  {
    for (int c = 0; c < (int)c_stop.size(); c++)
    {
      const class process_node &process_node =
          mpi_node_endpoints.get_process_node(r, c);

      const std::vector<class comm_packet_endpoint> &endpoint =
          process_node.endpoints_in_order();

      std::vector<int> derived_slice;

      /* only the receive-type endpoints are of interest here */
      for (int i = 0; i < (int)endpoint.size(); i++)
      {
        if (endpoint[i].endpoint_type == endpoint_enum::receive)
        {
          std::vector<int> sub_slice =
              cut_a_slice(endpoint[i].nar, c_stop, r_stop, x);

          std::copy(sub_slice.begin(), sub_slice.end(),
                    std::back_inserter(derived_slice));
        }
      }

      std::vector<int> goal_slice = correct_slice(c, c_stop, x);

      if (derived_slice != goal_slice)
        return false;
    }
  }

  return true;
}

/* ensure that every send has a matching receive with the same info */
bool check_packet_endpoints(const class mpi_node_endpoints &mpi_node_endpoints)
{
  /* these arrays will have the format:
     { sending linear index, receiving linear index, start, stop } */
  std::set<std::array<const int, 4>> receive_endpoint;

  std::vector<std::array<const int, 4>> send_endpoint;

  for (int r = 0; r < mpi_node_endpoints.n_tile_rows(); r++)
  {
    for (int c = 0; c < mpi_node_endpoints.n_tile_cols(); c++)
    {
      const class process_node &process_node =
          mpi_node_endpoints.get_process_node(r, c);

      const std::vector<class comm_packet_endpoint> &endpoint =
          process_node.endpoints_in_order();

      for (int i = 0; i < (int)endpoint.size(); i++)
      {
        const class node_and_range &nar = endpoint[i].nar;

        if (endpoint[i].endpoint_type == endpoint_enum::send)
        {
          // clang-format off
          const std::array< const int, 4 > array{ r * mpi_node_endpoints.n_tile_cols() + c,
                                                  nar.linear_index,
                                                  nar.start,
                                                  nar.stop };
          // clang-format on

          send_endpoint.emplace_back(std::move(array));
        }

        else if (endpoint[i].endpoint_type == endpoint_enum::receive)
        {
          // clang-format off
          std::array< const int, 4 > array{ nar.linear_index,
                                            r * mpi_node_endpoints.n_tile_cols() + c,
                                            nar.start,
                                            nar.stop };
          // clang-format on

          receive_endpoint.emplace(std::move(array));
        }
      }
    }
  }

  if (send_endpoint.size() != receive_endpoint.size())
    return false;

  for (int i = 0; i < (int)send_endpoint.size(); i++)
  {
    if (receive_endpoint.find(send_endpoint[i]) == receive_endpoint.end())
      return false;
  }

  return true;
}

TEST_CASE("mpi_endpoints", "[mpi]")
{
  /* testing parameters */
  int c_segments = 100;

  int interval_max = 100;

  std::random_device rdev;

  /* test data */
  std::vector<int> c_stop = segment_range(interval_max, c_segments, rdev);

  std::vector<int> r_stop = reverse_offset(c_stop);

  /* create object */
  class mpi_node_endpoints mpi_node_endpoints(std::move(r_stop),
                                              std::move(c_stop));

  SECTION("rowspace_intervals")
  {
    const std::vector<std::vector<class node_and_range>> row_space_intervals =
        mpi_node_endpoints.gen_row_space_intervals();

    // clang-format off
    bool pass = check_row_space_intervals( row_space_intervals,
                                           mpi_node_endpoints.get_c_stop(),
                                           mpi_node_endpoints.get_r_stop() );
    // clang-format on

    REQUIRE(pass == true);
  }

  SECTION("build_slices")
  {
    bool pass = check_slices(mpi_node_endpoints);

    REQUIRE(pass == true);
  }

  SECTION("check_endpoints")
  {
    bool pass = check_packet_endpoints(mpi_node_endpoints);

    REQUIRE(pass == true);
  }
}
