#include "mpi_endpoints.hpp"

wheel::wheel(int size)
    // clang-format off
  :
  size( size ),
  current_index( 0 )
// clang-format on
{
  return;
}

int wheel::spin()
{
  int n = current_index++;

  if (current_index == size)
    current_index = 0;

  return n;
}

node_and_range::node_and_range(int linear_index, int start, int stop)
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
comm_packet_endpoint::comm_packet_endpoint( endpoint_enum endpoint_type,
                                            class node_and_range &&nar )
  :
  endpoint_type( endpoint_type ),
  nar( nar )
// clang-format on
{
  return;
}

void process_node::add_endpoint(class comm_packet_endpoint &&item)
{
  endpoint.push_back(item);

  return;
}

const std::vector<std::vector<class node_and_range>>
mpi_node_endpoints::gen_row_space_intervals()
{
  std::vector<std::vector<class node_and_range>> v(c_stop.size());

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
void mpi_node_endpoints::gen_endpoints( const std::vector< std::vector< class node_and_range > >
                                        &row_space_intervals )
// clang-format on
{
  for (int c = 0; c < (int)row_space_intervals.size(); c++)
  {
    std::vector<class node_and_range> nar_vec = row_space_intervals[c];

    /* everything in this vector is needed by nodes: ( row, i )*/
    for (int j = 0; j < (int)nar_vec.size(); j++)
    {
      class node_and_range &nar = nar_vec[j];

      for (int r = 0; r < (int)r_stop.size(); r++)
      {
        /* construct the receive item */
        int receiving_linear_index = r * c_stop.size() + c;

        class process_node &node = process_node[receiving_linear_index];

        int from_linear_index = receiving_linear_index;

        if (nar.linear_index != r)
        {
          /* remember, in this context, nar.linear index refers to a row */
          from_linear_index =
              nar.linear_index * c_stop.size() + row_wheel[r].spin();
        }

        /* construct the receive item */
        class node_and_range n_move_2(from_linear_index, nar.start, nar.stop);

        class comm_packet_endpoint cpe(endpoint_enum::receive,
                                       std::move(n_move_2));

        node.add_endpoint(std::move(cpe));

        /* construct the corresponding send item */
        class process_node &node_0 = process_node[from_linear_index];

        class node_and_range n_move_3(receiving_linear_index, nar.start,
                                      nar.stop);

        class comm_packet_endpoint cpe_0(endpoint_enum::send,
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
  process_node.resize(r_stop.size() * c_stop.size());

  /* create row wheel */
  row_wheel.reserve(r_stop.size());

  for (int i = 0; i < (int)r_stop.size(); i++)
  {
    row_wheel.push_back(c_stop.size());
  }

  const std::vector<std::vector<class node_and_range>> row_space_intervals =
      gen_row_space_intervals();

  /* Captain! */
  /* generate endpoint lists for each process node */
  gen_endpoints(row_space_intervals);

  return;
}
