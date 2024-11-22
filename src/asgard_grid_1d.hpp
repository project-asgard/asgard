#pragma once

#include "asgard_fast_math.hpp"

namespace asgard
{
/*!
 * \brief Keeps track of the connectivity of the elements in the 1d hierarchy.
 *
 * Constructs a sparse matrix-like structure with row-compressed format and
 * ordered indexes within each row, so that the 1D connectivity can be verified
 * with a simple binary search.
 *
 * The advantage of the structure is to provide:
 * - easier check if two 1D cell are connected or not
 * - index of the connection, so operator coefficient matrices can be easily
 *   referenced
 */
class connect_1d
{
public:
  //! \brief Indicates whether to include same level edge neighbours
  enum class hierarchy : int
  {
    //! \brief Uses the volume connectivity
    volume = 0,
    //! \brief Uses the full volume and edge connectivity
    full,
    //! \brief Column transfomed volumes from diagonal (but not row transformed)
    col_volume,
    //! \brief Column transfomed full from triangular (but not row transformed)
    col_full,
  };
  //! Type tag for column transformations
  struct column_extended_hierarchy{};
  //! Type tag for column transformations
  static column_extended_hierarchy col_extend_hierarchy;
  //! Placeholder, empty connection
  connect_1d() : levels(0), rows(0) {}
  /*!
   *  \brief Constructor, makes the connectivity up to and including the given
   *         max-level.
   */
  connect_1d(int const max_level, hierarchy mode = hierarchy::full)
      : levels(max_level)
  {
    expect(mode == hierarchy::full or mode == hierarchy::volume);
    switch (mode)
    {
    case hierarchy::full:
      build_connections<hierarchy::full>();
      break;
    case hierarchy::volume:
      build_connections<hierarchy::volume>();
      break;
    default:
      // maybe redundant with the excpect above
      throw std::runtime_error("constructor for full or volume only");
      break;
    }
  }
  /*!
   * \brief Creates a new connectivity by setting up workspace for the
   *        hierarchical column transformation
   */
  connect_1d(connect_1d const &conn, column_extended_hierarchy)
    : levels(conn.levels), rows(std::max(1, 2 * conn.rows - 2))
  {
    if (rows == 1)
    {
      pntr = {0, 1};
      indx = {0, };
      return;
    }
    if (rows == 2)
    {
      pntr = {0, 2, 4};
      indx = {0, 1, 0, 1};
      return;
    }
    // trivial cases are done, working on extending the pattern
    pntr.reserve(rows + 1);
    for (int i = 0; i < 3; i++)
      pntr.push_back(conn.pntr[i]);
    int num = 1; // number of rows per level
    for (int l = 2; l <= levels; l++)
    {
      num *= 2;
      for (int r = num; r < 2 * num; r++)
      {
        int const row_length = conn.row_end(r) - conn.row_begin(r);
        pntr.push_back(pntr.back() + row_length); // 2 identical rows
        pntr.push_back(pntr.back() + row_length);
      }
    }
    // setting up the indexes, first two rows are dense
    indx.reserve(pntr.back());
    for (int r = 0; r < conn.rows; r++)
      indx.push_back(r);
    for (int r = 0; r < conn.rows; r++)
      indx.push_back(r);
    num = 1; // number of rows per level
    for (int l = 2; l <= levels; l++)
    {
      num *= 2;
      for (int r = num; r < 2 * num; r++)
      {
        for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
          indx.push_back(conn[j]);
        for (int j = conn.row_begin(r); j < conn.row_end(r); j++)
          indx.push_back(conn[j]);
      }
    }
  }
  /*!
   * \brief Creates a new connectivity matrix by expanding each element with
   *        a block of size (degree+1) by (degree+1)
   */
  connect_1d(connect_1d const &elem_connect, int degree)
      : levels(-1), rows(elem_connect.rows * (degree + 1)),
        pntr(rows + 1, 0), diag(rows)
  {
    int const block_rows = degree + 1;

    pntr[0] = 0;
    for (int row = 0; row < elem_connect.num_rows(); row++)
    {
      int elem_per_row = block_rows * (elem_connect.row_end(row) - elem_connect.row_begin(row));
      for (int j = 0; j < block_rows; j++)
        pntr[block_rows * row + j + 1] = pntr[block_rows * row + j] + elem_per_row;
    }

    // add the connectivity entries
    indx.reserve(pntr.back());
    for (int row = 0; row < elem_connect.num_rows(); row++)
    {
      for (int j = 0; j < block_rows; j++)
      {
        for (int col = elem_connect.row_begin(row); col < elem_connect.row_diag(row);
             col++)
          for (int k = 0; k < block_rows; k++)
            indx.push_back(block_rows * elem_connect[col] + k);

        // keep only one entry from the diagonal block
        for (int k = 0; k < j; k++)
          indx.push_back(block_rows * elem_connect[elem_connect.row_diag(row)] + k);
        diag[block_rows * row + j] = static_cast<int>(indx.size());
        indx.push_back(block_rows * row + j);
        for (int k = j + 1; k < block_rows; k++)
          indx.push_back(block_rows * elem_connect[elem_connect.row_diag(row)] + k);

        for (int col = elem_connect.row_diag(row) + 1; col < elem_connect.row_end(row);
             col++)
          for (int k = 0; k < block_rows; k++)
            indx.push_back(block_rows * elem_connect[col] + k);
      }
    }
  }

  int get_offset(int row, int col) const
  {
    // there is a potential for optimization here, look into it later
    // if we use the hierarchy with all elements connected by volume
    // the first two elements have lots of connection which slows the search
    // but the search result is trivial
    //if (row == 0)
    //  return col;
    //else if (row == 1)
    //  return rows + col;
    // if not on the first or second row, do binary search
    int sstart = pntr[row], send = pntr[row + 1] - 1;
    int current = (sstart + send) / 2;
    while (sstart <= send)
    {
      if (indx[current] < col)
      {
        sstart = current + 1;
      }
      else if (indx[current] > col)
      {
        send = current - 1;
      }
      else
      {
        return current;
      };
      current = (sstart + send) / 2;
    }
    return -1;
  }

  //! \brief Total number of connections (non-zeros).
  int num_connections() const { return static_cast<int>(indx.size()); }

  int num_rows() const { return rows; }

  int row_begin(int row) const { return pntr[row]; }

  int row_diag(int row) const { return diag[row]; }

  int row_end(int row) const { return pntr[row + 1]; }

  //! \brief Index at offset j.
  int operator[](int j) const { return indx[j]; }

  int max_loaded_level() const { return levels; }

  void print(std::ostream &os = std::cout) const // for debugging
  {
    for (int r = 0; r < rows; r++)
    {
      os << "row = " << std::setw(3) << r << ": "
         << std::setw(3) << indx[row_begin(r)];
      for (int j = row_begin(r) + 1; j < row_end(r); j++)
        os << " " << std::setw(3) << indx[j];
      os << '\n';
    }
    if (rows == 0)
    {
      os << "diag = <connect_1d pattern is empty> \n";
      return;
    }
    os << "diag = " << std::setw(3) << diag[0];
    for (int r = 1; r < rows; r++)
      os << " " << std::setw(3) << diag[r];
    os << '\n';
  }

protected:
  //! \brief Allows for different hierarchy modes with if-constexpr
  template<hierarchy mode>
  void build_connections()
  {
    rows = fm::ipow2(levels);
    pntr.resize(rows + 1, 0);
    indx.resize(2 * rows);
    diag.resize(rows);

    // compute the powers of two
    std::vector<int> cell_per_level(levels + 2, 1);
    for (size_t l = 2; l < cell_per_level.size(); l++)
      cell_per_level[l] = 2 * cell_per_level[l - 1];

    // first two cells are connected to everything
    pntr[1] = rows;
    pntr[2] = 2 * rows;
    for (int i = 0; i < rows; i++)
      indx[i] = i;
    for (int i = 0; i < rows; i++)
      indx[i + rows] = i;
    diag[0] = 0;
    diag[1] = 1 + rows;

    // for the remaining, loop level by level, cell by cell
    for (int l = 2; l < levels + 1; l++)
    {
      int level_size = cell_per_level[l]; // number of cells in this level

      // for each cell in this level, look at all cells connected
      // look at previous levels, this level, follow on levels

      // start with the first cell, on the left edge
      int i = level_size; // index of the first cell
      // always connected to cells 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      // look at cells above
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is volume-connected left-most cell
        indx.push_back(cell_per_level[upl]);
        // edge connected to the right-most cell
        if constexpr (mode == hierarchy::full)
          indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // look at this level
      diag[i] = static_cast<int>(indx.size());
      indx.push_back(i);
      if constexpr (mode == hierarchy::full)
      {
        indx.push_back(i + 1);
        // connect also to the right-most cell (periodic boundary)
        if (l > 2) // at level l = 2, i+1 is the right-most cell
          indx.push_back(cell_per_level[l + 1] - 1);
      }
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // connect to the first bunch of cell, i.e., with overlapping support
        // going on by 2, 4, 8 ... and one more for touching boundary
        // also connect to the right-most cell
        int lstart = cell_per_level[downl];
        for (int downp = 0; downp < cell_per_level[downl - l + 1]; downp++)
          indx.push_back(lstart + downp);
        if constexpr (mode == hierarchy::full)
        {
          indx.push_back(lstart + cell_per_level[downl - l + 1]);
          indx.push_back(cell_per_level[downl + 1] - 1);
        }
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with point

      // handle middle cells
      for (int p = 1; p < level_size - 1; p++)
      {
        i++;
        // always connected to the first two cells
        indx.push_back(0);
        indx.push_back(1);
        // ancestors on previous levels
        for (int upl = 2; upl < l; upl++)
        {
          int segment_size = cell_per_level[l - upl + 1];
          int ancestor     = p / segment_size;
          int edge         = p - ancestor * segment_size; // p % segment_size
          // if on the left edge of the ancestor
          if constexpr (mode == hierarchy::full)
            if (edge == 0)
              indx.push_back(cell_per_level[upl] + ancestor - 1);
          indx.push_back(cell_per_level[upl] + ancestor);
          // if on the right edge of the ancestor
          if constexpr (mode == hierarchy::full)
            if (edge == segment_size - 1)
              indx.push_back(cell_per_level[upl] + ancestor + 1);
        }
        // on this level
        if constexpr (mode == hierarchy::full)
          indx.push_back(i - 1);
        diag[i] = static_cast<int>(indx.size());
        indx.push_back(i);
        if constexpr (mode == hierarchy::full)
          indx.push_back(i + 1);
        // kids on further levels
        int left_kid = p; // initialize, will be updated on first iteration
        int num_kids = 1;
        for (int downl = l + 1; downl < levels + 1; downl++)
        {
          left_kid *= 2;
          num_kids *= 2;
          if constexpr (mode == hierarchy::full)
            indx.push_back(cell_per_level[downl] + left_kid - 1);
          for (int j = left_kid; j < left_kid + num_kids; j++)
            indx.push_back(cell_per_level[downl] + j);
          if constexpr (mode == hierarchy::full)
            indx.push_back(cell_per_level[downl] + left_kid + num_kids);
        }
        pntr[i + 1] = static_cast<int>(indx.size()); // done with cell i
      }

      // right edge cell
      i++;
      // always connected to 0 and 1
      indx.push_back(0);
      indx.push_back(1);
      for (int upl = 2; upl < l; upl++)
      {
        // edge cell is edge connected to left most cell on each level
        if constexpr (mode == hierarchy::full)
          indx.push_back(cell_per_level[upl]);
        // volume connected to the right most cell on each level
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // at this level
      if constexpr (mode == hierarchy::full)
      {
        // connect also to the left-most cell (periodic boundary)
        if (l > 2) // at level l = 2, left-most cell is i-1, don't double add
          indx.push_back(cell_per_level[l]);
        indx.push_back(i - 1);
      }
      diag[i] = static_cast<int>(indx.size());
      indx.push_back(i);
      // look at follow on levels
      for (int downl = l + 1; downl < levels + 1; downl++)
      {
        // left edge on the level
        if constexpr (mode == hierarchy::full)
          indx.push_back(cell_per_level[downl]);
        // get the last bunch of cells at the level
        int lend   = cell_per_level[downl + 1] - 1;
        int lbegin = cell_per_level[downl - l + 1] - 1;
        if constexpr (mode == hierarchy::full)
          lbegin += 1;
        for (int downp = lbegin; downp > -1; downp--)
          indx.push_back(lend - downp);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with the right edge

    } // done with level, move to the next level
  }   // done with the method

private:
  // pntr and indx form a sparse matrix (row-compressed format)
  // describing the connections between the indexes
  // diag[i] holds the offset of the diagonal entry, i.e., indx[diag[i]] = i
  //         it helps identify lower/upper triangular part of the pattern
  int levels;
  int rows;
  std::vector<int> pntr;
  std::vector<int> indx;
  std::vector<int> diag;
};

/*!
 * \brief Combines together a volume and an edge flux pattern
 */
struct connection_patterns
{
  //! construct patterns up to the given level
  explicit connection_patterns(int max_level)
  {
    // make the volume and edge connection patterns
    for (int i = 0; i < 2; i++)
      conns[i] = connect_1d(max_level, static_cast<connect_1d::hierarchy>(i));
    // make the column transform patterns
    for (int i = 2; i < 4; i++)
      conns[i] = connect_1d(conns[i - 2], connect_1d::col_extend_hierarchy);
  }
  //! return the corresponding connectivity pattern
  connect_1d const &operator() (connect_1d::hierarchy h) const
  {
    return conns[static_cast<int>(h)];
  }
  //! return the corresponding connectivity pattern
  connect_1d const &operator[] (connect_1d::hierarchy h) const
  {
    return conns[static_cast<int>(h)];
  }
  connect_1d const *get(connect_1d::hierarchy h) const
  {
      return &conns[static_cast<int>(h)];
  }
  std::array<connect_1d, 4> conns;
};

} // namespace asgard
