#include "mpi_endpoints.hpp"

circular_selector::circular_selector(int size)
    // clang-format off
  :
  size( size ),
  current_index( 0 )
// clang-format on
{
  return;
}

int circular_selector::spin()
{
  int n = current_index++;

  if (current_index == size)
    current_index = 0;

  return n;
}

mpi_node_and_range::mpi_node_and_range(int linear_index, int start, int stop)
    // clang-format off
  :
  linear_index( linear_index ),
  start( start ),
  stop( stop )
// clang-format on
{
  return;
}

// clang-format off
mpi_packet_endpoint::mpi_packet_endpoint( endpoint_enum endpoint_type,
                                          class mpi_node_and_range &&nar )
  :
  endpoint_type( endpoint_type ),
  nar( nar )
// clang-format on
{
  return;
}

void mpi_node_endpoint::add_endpoint(class mpi_packet_endpoint &&item)
{
  endpoint.push_back(item);

  return;
}

const std::vector<std::vector<class mpi_node_and_range>>
mpi_node_endpoints::gen_row_space_intervals()
{
  std::vector<std::vector<class mpi_node_and_range>> v(c_stop.size());

  int r_node = 0;

  int r_start = 0;

  int r_end = r_stop[r_node];

  int c_start = 0;

  int c_end = c_stop[0];

  for (int c = 0; c < (int)c_stop.size();)
  {
    if (c_end <= r_end)
    {
      v[c].emplace_back(r_node, c_start - r_start, c_end - r_start);

      c_start = c_end + 1;

      c_end = c_stop[++c];
    }

    else
    {
      v[c].emplace_back(r_node, c_start - r_start, r_end - r_start);

      r_start = r_end + 1;

      r_end = r_stop[++r_node];

      c_start = r_start;
    }
  }

  return v;
}

// clang-format off
void mpi_node_endpoints::gen_endpoints( const std::vector< std::vector< class mpi_node_and_range > >
                                        &row_space_intervals )
// clang-format on
{
  for (int c = 0; c < (int)row_space_intervals.size(); c++)
  {
    std::vector<class mpi_node_and_range> nar_vec = row_space_intervals[c];

    /* everything in this vector is needed by nodes: ( row, i )*/
    for (int j = 0; j < (int)nar_vec.size(); j++)
    {
      class mpi_node_and_range &nar = nar_vec[j];

      for (int r = 0; r < (int)r_stop.size(); r++)
      {
        /* construct the receive item */
        int receiving_linear_index = r * c_stop.size() + c;

        class mpi_node_endpoint &node = mpi_node_endpoint[receiving_linear_index];

        int from_linear_index = receiving_linear_index;

        if (nar.linear_index != r)
        {
          /* remember, in this context, nar.linear index refers to a row */
          from_linear_index =
              nar.linear_index * c_stop.size() + row_circular_selector[r].spin();
        }

        /* construct the receive item */
        class mpi_node_and_range n_move_2(from_linear_index, nar.start, nar.stop);

        class mpi_packet_endpoint cpe(endpoint_enum::receive,
                                       std::move(n_move_2));

        node.add_endpoint(std::move(cpe));

        /* construct the corresponding send item */
        class mpi_node_endpoint &node_0 = mpi_node_endpoint[from_linear_index];

        class mpi_node_and_range n_move_3(receiving_linear_index, nar.start,
                                      nar.stop);

        class mpi_packet_endpoint cpe_0(endpoint_enum::send,
                                         std::move(n_move_3));

        node_0.add_endpoint(std::move(cpe_0));
      }
    }
  }

  return;
}

// clang-format off
mpi_node_endpoints::mpi_node_endpoints( const std::vector< int > &&r_stop,
                                        const std::vector< int > &&c_stop )
  :
  r_stop( r_stop ),
  c_stop( c_stop )
// clang-format on
{
  /* total number of subgrids, each owned by a process node */
  mpi_node_endpoint.resize(r_stop.size() * c_stop.size());

  /* create row circular_selector */
  row_circular_selector.reserve(r_stop.size());

  for (int i = 0; i < (int)r_stop.size(); i++)
  {
    row_circular_selector.push_back(c_stop.size());
  }

  const std::vector<std::vector<class mpi_node_and_range>> row_space_intervals =
      gen_row_space_intervals();

  /* Captain! */
  /* generate endpoint lists for each process node */
  gen_endpoints(row_space_intervals);

  return;
}
