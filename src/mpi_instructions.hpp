/* have the function take in a distrubution plan */
#include <vector>

enum class mpi_message_enum
{
  send,
  receive
};

/* utility class for round robin selection */
class row_round_robin_wheel
{
public:
  row_round_robin_wheel(int size);

  int spin();

private:
  int const size;

  int current_index;
};

/* used to describe a range of elements and a node from which to obtain the
 * range */
class mpi_node_and_range
{
public:

  mpi_node_and_range(int linear_index, int start, int stop);

  int const linear_index;

  int const start;

  int const stop;
};

/* an mpi communication mpi_message can either be a send or a receive. */
class mpi_message
{
public:
  mpi_message(mpi_message_enum mpi_message_type, class mpi_node_and_range &nar);

  mpi_message_enum const mpi_message_type;

  mpi_node_and_range const nar;
};

/* partitions a matrix and assigns a process to each tile */
/* the rows and columns for tile( row, column ) owned by process( row, column )
   include rows: [ r_stop[ row - 1 ] + 1, r_stop[ row ] ] inclusive and
   columns: [ c_stop[ col - 1 ] + 1, c_stop[ col ] ] inclusive.
   The first row and column are assumed to start at 0. */

/* Assuming the partitioning above, generate_mpi_messages() creates a set of instructions for
   each process.*/

/* The communication for this module follows the access pattern for a
   matrix-vector multiply in a loop: Ax = b in which all tiles on row "i" have
   the same data. Specifically, any tile on tile row "i" contains a range of
   elements from the vector "x" corresponding to the rows of that tile. */

/* the vector is tile_columns elements long. Each one is a vector of intervals in row space */
std::vector<std::vector<mpi_node_and_range>>
generate_row_space_intervals( std::vector<int> const &r_stop, std::vector< int > const &c_stop );

/* each element of the vector is a nested vector specifying a mixed sequence of send and receive
   calls. The element at index "i" indicates the sequence for rank "i". If the send and receives
   are called in the order they appear, the system is guaranteed not to deadlock */
std::vector< std::vector< mpi_message > > const
generate_mpi_messages( const std::vector<int> &r_stop,
                       const std::vector<int> &c_stop );
