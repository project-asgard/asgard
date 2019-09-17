#pragma once

#include "chunk.hpp"
#include "distribution.hpp"
#include <vector>

/* utility class for round robin selection */
class round_robin_wheel
{
public:
  round_robin_wheel(int const size) : size(size), current_index(0) {}
  int spin();

private:
  int const size;
  int current_index;
};

/* a message can either be a send or a receive  */
/* target is the sender rank for a receive, and receive rank for send  */
/* the range describes the global indices (inclusive) that will be transmitted
 */
enum class message_direction
{
  send,
  receive
};

struct message
{
  message(message_direction const message_dir, int const target, limits<> range)
      : message_dir(message_dir), target(target), range(range)
  {}
  message_direction const message_dir;
  int const target;
  limits<> range;
};

/* the rows and columns for subgrid( row, column ) owned by rank( row, column )
   include rows: [ row_boundaries[ row - 1 ] + 1, row_boundaries[ row ] ]
   inclusive and columns: [ column_boundaries[ col - 1 ] + 1, column_boundaries[
   col ] ] inclusive. The first row and column are assumed to start at 0. */

/* Assuming the partitioning above, generate_messages() creates a set of
   instructions for each rank.*/

/* The communication for this module follows the access pattern for a
   matrix-vector multiply in a loop: Ax = b in which all subgrid on row "i" have
   the same data. */

/* the vector is num_subgrid_columns in length. Element "x" in this vector
 * describes the subgrid rows holding data that members of subgrid column "x"
 * need to receive, as well as the global indices of that data in the solution
 * vector  */
using row_to_range = std::map<int, limits<>>;
std::vector<row_to_range>
generate_row_intervals(std::vector<int> const &row_boundaries,
                       std::vector<int> const &column_boundaries);

std::vector<std::vector<message>> const
intervals_to_messages(std::vector<row_to_range> const &row_intervals,
                      std::vector<int> const &row_boundaries,
                      std::vector<int> const &column_boundaries);

/* given a distribution plan, map each rank to a list of messages */
/* index "x" of this vector contains the messages that must be transmitted
 * for rank "x" */

/* if the messages are invoked in order, they are guaranteed not
 * to produce a deadlock */
std::vector<std::vector<message>> const
generate_messages(distribution_plan const &plan);
