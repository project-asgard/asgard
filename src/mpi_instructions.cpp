
#include <map>

#include "chunk.hpp"
#include "mpi_instructions.hpp"

int round_robin_wheel::spin()
{
  int const n = current_index++;

  if (current_index == size)
    current_index = 0;

  return n;
}

std::vector<row_to_range>
generate_row_intervals(std::vector<int> const &row_boundaries,
                       std::vector<int> const &column_boundaries)
{
  // contains an element for each subgrid column describing
  // the subgrid rows, and associated ranges, that the column
  // members will need information from
  std::vector<row_to_range> row_intervals(column_boundaries.size());

  // start at the first row and column interval
  // col_start is the first index in this column interval
  int col_start = 0;
  for (int c = 0; c < static_cast<int>(column_boundaries.size()); ++c)
  {
    int row_start = 0;
    // the stop vectors represent the end of a range
    int const column_end = column_boundaries[c];
    for (int r = 0; r < static_cast<int>(row_boundaries.size()); ++r)
    {
      int const row_end = row_boundaries[r];

      // if the row interval falls within the column interval
      if ((col_start >= row_start && col_start <= row_end) ||
          (row_start >= col_start && row_start <= column_end))
      {
        // emplace the section of the row interval that falls within the column
        // interval
        row_intervals[c].emplace(r, limits<>(std::max(row_start, col_start),
                                             std::min(row_end, column_end)));
      }

      // the beginning of the next interval is one more than the end of the
      // previous
      row_start = row_end + 1;
    }
    col_start = column_end + 1;
  }

  return row_intervals;
}

std::vector<std::vector<message>> const
intervals_to_messages(std::vector<row_to_range> const &row_intervals,
                      std::vector<int> const &row_boundaries,
                      std::vector<int> const &column_boundaries)
{
  assert(column_boundaries.size() == row_boundaries.size());
  assert(row_intervals.size() == column_boundaries.size());

  /* initialize a round robin selector for each row */
  std::vector<round_robin_wheel> row_round_robin_wheels;
  for (int i = 0; i < static_cast<int>(row_boundaries.size()); ++i)
  {
    row_round_robin_wheels.emplace_back(column_boundaries.size());
  }

  /* this vector contains lists of messages indexed by rank */
  std::vector<std::vector<message>> messages(row_boundaries.size() *
                                             column_boundaries.size());

  /* iterate over each subgrid column's input requirements */
  for (int c = 0; c < static_cast<int>(row_intervals.size()); c++)
  {
    /* interval map describes the subgrid row each column member will need
     * to communicate with, as well as the solution vector ranges needed
     * from each. these requirements are the same for every column member */
    row_to_range const interval_map = row_intervals[c];
    for (auto const &[row, limits] : interval_map)
    {
      /* iterate every rank in the subgrid column */
      for (int r = 0; r < static_cast<int>(row_boundaries.size()); ++r)
      {
        /* construct the receive item */
        int const receiver_rank = r * column_boundaries.size() + c;

        /* if receiver_rank has the data it needs locally, it will copy from its
         * own output otherwise, use round robin wheel to select a sender from
         * another row - every member of the row has the same data */
        int const sender_rank = [row = row, r, receiver_rank,
                                 &column_boundaries,
                                 &wheel = row_round_robin_wheels[row]]() {
          if (row == r)
          {
            return receiver_rank;
          }
          return static_cast<int>(row * column_boundaries.size() +
                                  wheel.spin());
        }();

        /* add message to the receiver's message list */
        message const incoming_message(message_direction::receive, sender_rank,
                                       limits);
        messages[receiver_rank].push_back(incoming_message);

        /* construct and enqeue the corresponding send item */
        message const outgoing_message(message_direction::send, receiver_rank,
                                       limits);
        messages[sender_rank].push_back(outgoing_message);
      }
    }
  }

  return messages;
}

std::vector<std::vector<message>> const
generate_messages(distribution_plan const plan)
{
  /* first, determine the subgrid tiling for this plan */
  std::vector<int> row_boundaries;
  std::vector<int> col_boundaries;

  auto const num_cols = get_num_subgrid_cols(plan.size());
  assert(plan.size() % num_cols == 0);
  auto const num_rows = static_cast<int>(plan.size()) / num_cols;

  for (int i = 0; i < num_rows; ++i)
  {
    element_subgrid const &grid = plan.at(i * num_cols);
    row_boundaries.push_back(grid.row_stop);
  }

  for (int i = 0; i < num_cols; ++i)
  {
    element_subgrid const &grid = plan.at(i);
    col_boundaries.push_back(grid.col_stop);
  }

  /* describe the rows/ranges each column needs to communicate with */
  auto const row_intervals =
      generate_row_intervals(row_boundaries, col_boundaries);
  /* finally, build message list */
  auto const messages =
      intervals_to_messages(row_intervals, row_boundaries, col_boundaries);

  return messages;
}
