#pragma once

#include "elements.hpp"

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
  enum same_level { level_edge_include, level_edge_skip };
  /*!
   * \brief Type tag indicating a pattern that includes only the cells on the
   * same level.
   */
  struct tag_level_edge_only{};
  //! \brief Instance of the tag for easy use.
  static tag_level_edge_only level_edge_only;
  /*!
   *  \brief Constructor, makes the connectivity up to and including the given
   *         max-level.
   */
  connect_1d(int const max_level, same_level neighbor = level_edge_include)
      : levels(max_level), rows(fm::two_raised_to(levels)), pntr(rows + 1, 0),
        indx(2 * rows), diag(rows)
  {
    std::vector<int> cell_per_level(levels + 2, 1);
    for (int l = 2; l < levels + 2; l++)
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
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // look at this level
      diag[i] = static_cast<int>(indx.size());
      indx.push_back(i);
      if (neighbor == level_edge_include)
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
        for (int downp = 0; downp < cell_per_level[downl - l + 1] + 1; downp++)
          indx.push_back(lstart + downp);
        indx.push_back(cell_per_level[downl + 1] - 1);
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
          if (edge == 0)
            indx.push_back(cell_per_level[upl] + ancestor - 1);
          indx.push_back(cell_per_level[upl] + ancestor);
          // if on the right edge of the ancestor
          if (edge == segment_size - 1)
            indx.push_back(cell_per_level[upl] + ancestor + 1);
        }
        // on this level
        if (neighbor == level_edge_include)
          indx.push_back(i - 1);
        diag[i] = static_cast<int>(indx.size());
        indx.push_back(i);
        if (neighbor == level_edge_include)
          indx.push_back(i + 1);
        // kids on further levels
        int left_kid = p; // initialize, will be updated on first iteration
        int num_kids = 1;
        for (int downl = l + 1; downl < levels + 1; downl++)
        {
          left_kid *= 2;
          num_kids *= 2;
          for (int j = left_kid - 1; j < left_kid + num_kids + 1; j++)
            indx.push_back(cell_per_level[downl] + j);
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
        // edge cell is connected to both edge cells on each level
        indx.push_back(cell_per_level[upl]);
        indx.push_back(cell_per_level[upl + 1] - 1);
      }
      // at this level
      if (neighbor == level_edge_include)
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
        indx.push_back(cell_per_level[downl]);
        // get the last bunch of cells at the level
        int lend = cell_per_level[downl + 1] - 1;
        for (int downp = cell_per_level[downl - l + 1]; downp > -1; downp--)
          indx.push_back(lend - downp);
      }
      pntr[i + 1] = static_cast<int>(indx.size()); // done with the right edge
    } // done with level, move to the next level
  }   // close the constructor

  /*!
   * \brief Construct a pattern including only the elements on the same level
   *        and connected by the edge (also includes the self-connection).
   *
   * Currently not used, but maybe modified and used in the future.
   */
  connect_1d(int const max_level, tag_level_edge_only)
      : levels(max_level), rows(1 << levels), pntr(rows + 1, 0),
        indx(2), diag(rows)
  {
    std::vector<int> cell_per_level(levels + 2, 1);
    for (int l = 2; l < levels + 2; l++)
      cell_per_level[l] = 2 * cell_per_level[l - 1];

    // first two cells are connected only to themselves
    pntr[1] = 1;
    pntr[2] = 2;
    indx[0] = 0;
    indx[1] = 1;
    diag[0] = 0;
    diag[1] = 1;

    // for the remaining, loop level by level, cell by cell
    for (int l = 2; l < levels + 1; l++)
    {
      int level_size = cell_per_level[l]; // number of cells in this level

      // for each cell in this level, look at all cells connected
      // look at previous levels, this level, follow on levels

      // start with the first cell, on the left edge
      int i = level_size; // index of the first cell
      diag[i] = static_cast<int>(indx.size()); // self-connect
      indx.push_back(i);
      indx.push_back(i + 1); // connect to one to the left
      if (l > 2) // at l==2, the i+1 cell is the right-most cell
        indx.push_back(cell_per_level[l + 1] - 1); // connect to right edge
      pntr[i + 1] = static_cast<int>(indx.size()); // done with point

      // handle middle cells
      for (int p = 1; p < level_size - 1; p++)
      {
        i++;
        indx.push_back(i - 1); // left-connect
        diag[i] = static_cast<int>(indx.size());
        indx.push_back(i); // self-connect
        indx.push_back(i + 1); // right-connect
        pntr[i + 1] = static_cast<int>(indx.size()); // done with cell i
      }

      // right edge cell
      i++;
      // connect also to the left-most cell (periodic boundary)
      if (l > 2) // at l==2  the i-1 cell is the left-most cell
        indx.push_back(cell_per_level[l]);
      indx.push_back(i - 1); // connect to the left-cell
      diag[i] = static_cast<int>(indx.size()); // self-connect
      indx.push_back(i);
      pntr[i + 1] = static_cast<int>(indx.size()); // done with the right edge
    } // done with level, move to the next level
  }   // close the constructor

  /*!
   * \brief Creates a new connectivity matrix by expanding each element with
   *        a block of size (porder+1) by (porder+1)
   */
  connect_1d(connect_1d const &elem_connect, int porder)
    : levels(-1), rows(elem_connect.rows * (porder+1)),
      pntr(rows + 1, 0), diag(rows)
  {
    int const block_rows = porder + 1;
    pntr[0] = 0;
    for(int row=0; row<elem_connect.num_rows(); row++)
    {
      int elem_per_row = block_rows * (elem_connect.row_end(row) - elem_connect.row_begin(row));
      for(int j=0; j<block_rows; j++)
        pntr[block_rows * row + j + 1] = pntr[block_rows * row + j] + elem_per_row;
    }

    // add the connectivity entries
    indx.reserve(pntr.back());
    for(int row=0; row<elem_connect.num_rows(); row++)
    {
      for(int j=0; j<block_rows; j++)
      {
        for(int col=elem_connect.row_begin(row); col<elem_connect.row_diag(row);
            col++)
          for(int k=0; k<block_rows; k++)
            indx.push_back(block_rows * elem_connect[col] + k);

        // keep only one entry from the diagonal block
        for(int k=0; k<j; k++)
          indx.push_back(block_rows * elem_connect[elem_connect.row_diag(row)] + k);
        diag[block_rows * row + j] = static_cast<int>(indx.size());
        indx.push_back(block_rows * row + j);
        for(int k=j+1; k<block_rows; k++)
          indx.push_back(block_rows * elem_connect[elem_connect.row_diag(row)] + k);

        for(int col=elem_connect.row_diag(row)+1; col<elem_connect.row_end(row);
            col++)
          for(int k=0; k<block_rows; k++)
            indx.push_back(block_rows * elem_connect[col] + k);
      }
    }
  }

  int get_offset(int row, int col) const
  {
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

  void dump() const // for debugging
  {
    std::cerr << "dumping connectivity to std::cerr\n";
    for(int r=0; r < num_rows(); r++)
    {
      for(int j=row_begin(r); j<row_end(r); j++)
        std::cerr << indx[j] << "  ";
      std::cerr << '\n';
    }
    std::cerr << "diag = ";
    for(int r=0; r < num_rows(); r++)
      std::cerr << diag[r] << "  ";
    std::cerr << "\n";
    std::cerr << " ------------------ \n";
  }

private:
  // pntr and indx form a sparse matrix (row-compressed format)
  // describing the connections between the indexes
  // diag[i] holds the offset of the diagonal entry, i.e., indx[diag[i]] = i
  //         it helps identify lowe/upper triangular part of the pattern
  int levels;
  int rows;
  std::vector<int> pntr;
  std::vector<int> indx;
  std::vector<int> diag;
};

} // namespace asgard
