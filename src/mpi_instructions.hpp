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

/* contains a list of instructions to execute an ordered, mixed sequence of send
   and receive calls for an mpi rank to follow. If they are called in the order
   they appear in the list, the entire system is guaranteed not to deadlock */
class mpi_instruction
{
  /* functions */
public:
  mpi_instruction() { return; };

  void queue_mpi_message(mpi_message &item);

  std::vector<mpi_message> const &mpi_messages_in_order() const
  {
    return mpi_messages;
  };

  /* variables */
private:
  std::vector<mpi_message> mpi_messages;
};

/* partitions a matrix and assigns a process to each tile */
/* the rows and columns for tile( row, column ) owned by node( row, column )
   include rows: [ r_stop[ row - 1 ] + 1, r_stop[ row ] ] inclusive and
   columns: [ c_stop[ col - 1 ] + 1, c_stop[ col ] ] inclusive.
   The first row and column are assumed to start at 0. */

/* Assuming the partitioning above, this class creates a set of instructions for
   each process.*/

/* The communication for this module follows the access pattern for a
   matrix-vector multiply in a loop: Ax = b in which all tiles on row "i" have
   the same data. Specifically, any tile on tile row "i" contains a range of
   elements from the vector "x" corresponding to the rows of that tile. */
class mpi_instructions
{
  /* functions */
public:
  mpi_instructions(std::vector<int> const &&r_stop,
                   std::vector<int> const &&c_stop);

  mpi_instruction const &get_mpi_instructions(int linear_index) const
  {
    return mpi_instructions_vector[linear_index];
  };

  mpi_instruction const &get_mpi_instructions(int row, int col) const
  {
    return mpi_instructions_vector[row * c_stop.size() + col];
  };

  std::vector<int> const &get_c_stop() const { return c_stop; };

  std::vector<int> const &get_r_stop() const { return r_stop; };

  int n_tile_rows() const { return r_stop.size(); };

  int n_tile_cols() const { return c_stop.size(); };

  std::vector<std::vector<mpi_node_and_range>> const
  gen_row_space_intervals();

  /* functions */
private:
  void gen_mpi_messages(std::vector<std::vector<mpi_node_and_range>> const &row_space_intervals);

  /* variables */
private:
  /* each rank has an mpi_instruction object in this vector */
  std::vector<mpi_instruction> mpi_instructions_vector;

  std::vector<row_round_robin_wheel> row_row_round_robin_wheel;

  std::vector<int> const r_stop;

  std::vector<int> const c_stop;
};

std::vector<std::vector<mpi_node_and_range>>
generate_row_space_intervals( std::vector<int> const &r_stop, std::vector< int > const &c_stop );

std::vector< std::vector< mpi_message > > const
generate_mpi_messages( const std::vector<int> &r_stop,
                       const std::vector<int> &c_stop );
