/*

Proposed changes to go here:

1. Rename circular_selector to row_round_robin_wheel
2. Turn clang-format "on" in all instances
3. Comment the gen_row_space() function
4. Get rid of move semantics 
5. Flatten out any classes you can

*/
#include <vector>

/* a communication is characterized by two endpoints */
enum class endpoint_enum
{
  send,
  receive
};

/* utility class for round robin selection */
class circular_selector
{
public:
  circular_selector(int size);

  int spin();

private:
  int size;

  int current_index;
};

/* used to describe a range of elements and a node from which to obtain the
 * range */
class mpi_node_and_range
{
public:
  mpi_node_and_range(int linear_index, int start, int stop);

  const int linear_index;

  const int start;

  const int stop;
};

/* an mpi communication endpoint can either be a send or a receive. */
class mpi_packet_endpoint
{
public:
  // clang-format off
    mpi_packet_endpoint( endpoint_enum endpoint_type,
                          class mpi_node_and_range &&nar );
  // clang-format on

  const endpoint_enum endpoint_type;

  const class mpi_node_and_range nar;
};

/* contains all communication endpoints for a process in an ordered list. If
   sends and receives are called in the order they appear, the communication is
   guaranteed not to deadlock */
class mpi_node_endpoint
{
  /* functions */
public:
  mpi_node_endpoint() { return; };

  void add_endpoint(class mpi_packet_endpoint &&item);

  const std::vector<class mpi_packet_endpoint> &endpoints_in_order() const
  {
    return endpoint;
  };

  /* variables */
private:
  std::vector<class mpi_packet_endpoint> endpoint;
};

/* partitions a matrix and assigns a process to each tile */
/* the rows and columns for tile( row, column ) owned by node( row, column )
   include rows: [ r_stop[ row - 1 ] + 1, r_stop[ row ] ] inclusive and
   columns: [ c_stop[ col - 1 ] + 1, c_stop[ col ] ] inclusive.
   The first row and column are assumed to start at 0. */
/* Assuming the partitioning above, this class returns a process node for each
 * process.*/
/* The communication for this module follows the access pattern for a
   matrix-vector multiply in a loop: Ax = b in which all tiles on row "i" have
   the same data. Specifically, any tile on tile row "i" contains a range of
   elements from the vector "x" corresponding to the rows of that tile. */
class mpi_node_endpoints
{
  /* functions */
public:
  mpi_node_endpoints(const std::vector<int> &&r_stop,
                     const std::vector<int> &&c_stop);

  const class mpi_node_endpoint &get_mpi_node_endpoint(int linear_index) const
  {
    return mpi_node_endpoint[linear_index];
  };

  const class mpi_node_endpoint &get_mpi_node_endpoint(int row, int col) const
  {
    return mpi_node_endpoint[row * c_stop.size() + col];
  };

  const std::vector<int> &get_c_stop() const { return c_stop; };

  const std::vector<int> &get_r_stop() const { return r_stop; };

  int n_tile_rows() const { return r_stop.size(); };

  int n_tile_cols() const { return c_stop.size(); };

  const std::vector<std::vector<class mpi_node_and_range>>
  gen_row_space_intervals();

  /* functions */
private:
  void gen_endpoints(const std::vector<std::vector<class mpi_node_and_range>>
                         &row_space_intervals);

  /* variables */
private:
  /* each ( process, tile ) pair has a corresponding node */
  std::vector<class mpi_node_endpoint> mpi_node_endpoint;

  std::vector<class circular_selector> row_circular_selector;

  const std::vector<int> r_stop;

  const std::vector<int> c_stop;
};
