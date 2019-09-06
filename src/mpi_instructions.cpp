#include "mpi_instructions.hpp"
#include <iostream>

row_round_robin_wheel::row_round_robin_wheel(int size)
    : size(size), current_index(0)
{
  return;
}

int row_round_robin_wheel::spin()
{
  int n = current_index++;

  if (current_index == size)
    current_index = 0;

  return n;
}

mpi_node_and_range::mpi_node_and_range(int linear_index, int start, int stop)
    : linear_index(linear_index), start(start), stop(stop)
{
  return;
}

mpi_message::mpi_message(mpi_message_enum mpi_message_type,
                         class mpi_node_and_range &nar)
    : mpi_message_type(mpi_message_type), nar(nar)
{
  return;
}

void mpi_instruction::queue_mpi_message(class mpi_message &item)
{
  mpi_message.push_back(item);

  return;
}

/* this function describes each column interval as a combination of row
 * intervals */
const std::vector<std::vector<class mpi_node_and_range>>
mpi_instructions::gen_row_space_intervals()
{
  /* v is the row_space_intervals vector. It contains an element for each column
   * section */
  std::vector<std::vector<class mpi_node_and_range>> row_space_intervals(c_stop.size());

  /* start at the first column interval */
  int c_start = 0;

  int c_end = c_stop[0];

  /* begin describing its range with the first row interval */
  int r_node = 0;

  int r_start = 0;

  int r_end = r_stop[0];

  /* iterate over every column interval */
  for (int c = 0; c < (int)c_stop.size();)
  {
    /* if you have described the entire column interval and not exhausted the
       range of the current row interval, increment the pointer to the next
       column interval to be described */
    if (c_end <= r_end)
    {
      row_space_intervals[c].emplace_back(r_node, c_start - r_start, c_end - r_start);

      c_start = c_end + 1;

      if( c_end == r_end )
      {
        r_start = r_end + 1;

        r_end = r_stop[ ++r_node ];
      }

      c_end = c_stop[++c];
    }

    /* if you exhaust the range of the current row interval before describing
       the range of the current column interval, move on to the next row
       interval */
    else
    {
      row_space_intervals[c].emplace_back(r_node, c_start - r_start, r_end - r_start);

      r_start = r_end + 1;

      r_end = r_stop[++r_node];

      c_start = r_start;
    }
  }
  /* test code */
  std::cout << "r_stop: " << std::endl;
  for( int i = 0; i < r_stop.size(); i++ )
  {
    std::cout << " linear_index " << i << ": stop: " << r_stop[ i ] << std::endl;
  }
  std::cout << "c_stop: " << std::endl;
  for( int i = 0; i < c_stop.size(); i++ )
  {
    std::cout << " interval " << i << ": stop: " << c_stop[ i ] << std::endl;
  }

  std::cout << "row_space_intervals:" << std::endl;

  for( int i = 0; i < row_space_intervals.size(); i++ )
  {
    std::cout << "interval: " << i << std::endl;

    for( int j = 0; j < row_space_intervals[ i ].size(); j++ )
    {
      class mpi_node_and_range &nar = row_space_intervals[ i ][ j ];

      std::cout << "linear_index: " << nar.linear_index 
                << " start: " << nar.start
                << " stop: " << nar.stop
      << " global start: " 
      << nar.start + ( nar.linear_index == 0 ? 0 : r_stop[ nar.linear_index - 1 ] + 1)
      << " global stop: " 
      << nar.stop + ( nar.linear_index == 0 ? 0 : r_stop[ nar.linear_index - 1 ] + 1)
      << std::endl;
    }
    std::cout << std::endl;
  }
  /* end test code */

  return row_space_intervals;
}

/* take the output of gen_row_space_intervals() and construct mpi messages to
   move the specified ranges to the nodes they need to get to */
void mpi_instructions::gen_mpi_messages(
    const std::vector<std::vector<class mpi_node_and_range>>
        &row_space_intervals)
{
  /* iterate over every column space interval */
  for (int c = 0; c < (int)row_space_intervals.size(); c++)
  {
    /* this vector contains a set of row intervals and row interval
       sub-intervals that equal column interval "c" */
    std::vector<class mpi_node_and_range> nar_vec = row_space_intervals[c];

    /* every node in the same tile column needs the data described by the vector
     * above */
    for (int j = 0; j < (int)nar_vec.size(); j++)
    {
      /* the sub-range described by the node and range below has a corresponding
         send/receive pair */
      /* the linear_index field of this object indicates a tile row - each
         element in the tile row has the same data */
      class mpi_node_and_range &nar = nar_vec[j];

      /* iterate every node in the tile column */
      for (int r = 0; r < (int)r_stop.size(); r++)
      {
        /* construct the receive item */
        /* receiving_linear_index is the node that will be receiving this range
         */
        int receiving_linear_index = r * c_stop.size() + c;

        class mpi_instruction &receiving_node =
            mpi_instruction[receiving_linear_index];

        /* from_linear index is the node that will be sending the range to
           receiving_linear_index*/
        /* by default, assume that this node already has the data */
        int from_linear_index = receiving_linear_index;

        /* if it does not, spin the round robin row wheel to select a node from
         */
        /* all nodes in the same row have the data it needs, so round robin
           selection of a node from the correct row ensures balanced send
           operations */
        if (nar.linear_index != r)
        {
          from_linear_index = nar.linear_index * c_stop.size() +
                              row_row_round_robin_wheel[r].spin();
        }

        /* now that the correct sender has been determined, create a
           node_and_range from the sender and add it to the receiver's list of
           send/receive items */
        class mpi_node_and_range incoming_range(from_linear_index, nar.start,
                                                nar.stop);

        class mpi_message incoming_message(mpi_message_enum::receive,
                                           incoming_range);

        receiving_node.queue_mpi_message(incoming_message);

        /* construct the corresponding send item */
        class mpi_instruction &sending_node =
            mpi_instruction[from_linear_index];

        class mpi_node_and_range outgoing_range(receiving_linear_index,
                                                nar.start, nar.stop);

        class mpi_message outgoing_message(mpi_message_enum::send,
                                           outgoing_range);

        sending_node.queue_mpi_message(outgoing_message);
      }
    }
  }

  return;
}

mpi_instructions::mpi_instructions(const std::vector<int> &&r_stop,
                                   const std::vector<int> &&c_stop)
    : r_stop(r_stop), c_stop(c_stop)
{
  /* total number of subgrids, each owned by a process node */
  mpi_instruction.resize(r_stop.size() * c_stop.size());

  /* create row row_round_robin_wheel */
  row_row_round_robin_wheel.reserve(r_stop.size());

  for (int i = 0; i < (int)r_stop.size(); i++)
  {
    row_row_round_robin_wheel.push_back(c_stop.size());
  }

  const std::vector<std::vector<class mpi_node_and_range>> row_space_intervals =
      gen_row_space_intervals();

  /* generate mpi_message lists for each process node */
  gen_mpi_messages(row_space_intervals);

  return;
}
