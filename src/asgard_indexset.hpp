#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>

#include "tools.hpp"
#include "asgard_grid_1d.hpp"

namespace asgard
{

/*!
 * \brief Helper wrapper for data that will be organized in two dimensional form
 *
 * See the vector2d and span2d that derive from this class,
 * the purpose here is to reduce retyping of the same code.
 */
template<typename T, typename data_container>
class organize2d
{
public:
  //! \brief Virtual destructor.
  virtual ~organize2d() = default;

  //! \brief Returns the vector stride.
  int64_t stride() const { return stride_; }
  //! \brief Returns the vector stride.
  int64_t num_strips() const { return num_strips_; }
  //! \brief Returns the total number of entries.
  int64_t total_size() const { return stride_ * num_strips_; }
  //! \brief Returns true if empty.
  bool empty() const { return (num_strips_ == 0); }

  //! \brief Return pointer to the i-th strip.
  T* operator[] (int64_t i) { return data_[i * stride_]; }
  //! \brief Return const-pointer to the i-th strip.
  T const * operator[] (int64_t i) const { return data_[i * stride_]; }

protected:
  //! \brief Constructor, not intended for public use.
  organize2d(int64_t stride, int64_t num_strips)
    : stride_(stride), num_strips_(num_strips)
  {}

  int64_t stride_, num_strips_;
  data_container data_;
};

/*!
 * \brief Wrapper around std::vector, but providing 2d organization of the data.
 *
 * The data is divided into contiguous strips of fixed size.
 * The class provides easy access to individual strips, without the tedious
 * i * stride + j notation and with easy check for sizes.
 *
 * Note: while this is similar to a matrix, there is no notion of rows or column
 * and the data is not meant to represent a linear transformation.
 *
 * Allows access to the stride(), num_strips(), check for empty()
 * and access to alias to any strip with the [] operator.
 */
template<typename T>
class vector2d : public organize2d<T, std::vector<T>>
{
public:
  //! \brief Make an empty vector
  vector2d() : organize2d<T, std::vector<T>>::organize2d(0, 0) {}
  //! \brief Make a vector with the given dimensions, initialize to 0.
  vector2d(int64_t stride, int64_t num_strips)
    : organize2d<T, std::vector<T>>::organize2d(stride, num_strips)
  {
    this->data_ = std::vector<T>(stride * num_strips);
  }
  //! \brief Append to the end of the vector, assuming one strip of entries.
  void append(T const *p)
  {
    this->data_.insert(this->data_.end(), p, p + this->stride);
    this->num_strips_ += 1;
  }
  //! \brief Append to the end of the vector.
  void append(std::vector<T> const & p)
  {
    expect(static_cast<int64_t>(p.size()) % this->stride == 0);
    this->data_.insert(this->data_.end(), p.begin(), p.end());
    this->num_strips_ += static_cast<int64_t>(p.size()) / this->stride;
  }
};
//! \brief Non-owning version of vector2d.
template<typename T>
class span2d : public organize2d<T, T*>
{
public:
  //! \brief Make an empty vector
  span2d() : organize2d<T, T*>::organize2d(0, 0)
  {
    this->data_ = nullptr;
  }
  //! \brief Make a vector with the given dimensions, initialize to 0.
  span2d(int64_t stride, int64_t num_strips, T* data)
    : organize2d<T, T*>::organize2d(stride, num_strips)
  {
    this->data_ = data;
  }
};

/*!
 * \brief Contains a set of sorted multi-indexes
 *
 * For the given number of dimensions, the indexes are sorted
 * in lexicographical order with the last dimension changing the fastest.
 *
 * The purpose of this set is to establish an order between the multi-indexes
 * and the discretization cells that will be used to index the global degrees
 * of freedom.
 */
class indexset {
public:
  //! \brief Creates an empty set.
  indexset() : num_dimensions_(0), num_indexes_(0) {}
  //! \brief Creates a new set from a vector of sorted indexes.
  indexset(int num_dimensions, std::vector<int> &&indexes)
    : num_dimensions_(num_dimensions), indexes_(std::move(indexes))
  {
    expect(indexes.size() % num_dimensions_ == 0);
    num_indexes_ = static_cast<int>(indexes_.size() / num_dimensions_);
  }

  //! \brief Returns the number of stored multi-indexes.
  int num_indexes() const { return num_indexes_; }
  //! \brief Returns the number of dimensions.
  int num_dimensions() const { return num_dimensions_; }
  //! \brief Total number of integer entries.
  size_t size() const { return indexes_.size(); }
  //! \brief Returns true if the number of indexes is zero.
  bool empty() const { return (num_indexes() == 0); }

  //! \brief Get the i-th index of the lexicographical order.
  const int* operator[] (int i) const
  {
    return &indexes_[i * num_dimensions_];
  }

  //! \brief Find the index in the sorted list, returns -1 if missing
  int find(const int *idx) const
  {
    int first = 0, last = num_indexes_ - 1;
    int current = (first + last) / 2;
    while (first <= last)
    {
      match cmp = compare(current, idx);
      if (cmp == before_current)
      {
        last = current -1;
      }
      else if (cmp == after_current)
      {
        first = current +1;
      }
      else // match_found
      {
        return current;
      }
      current = (first + last) / 2;
    }
    return -1;
  }
  //! \brief Overload for std::vector
  int find(std::vector<int> const &idx) const
  {
    expect(num_dimensions_ == static_cast<int>(idx.size()));
    return find(idx.data());
  }

protected:
  //! \brief Result of a comparison
  enum match
  {
    before_current, match_found, after_current
  };
  //! \brief Compare the multi-index to the one at the position current.
  match compare(int current, int const *b) const
  {
    int const *a = (*this)[current];
    for(int j=0; j<num_dimensions_; j++) {
      if (a[j] < b[j]) return after_current;
      if (a[j] > b[j]) return before_current;
    }
    return match_found;
  }

private:
  int num_dimensions_;
  int num_indexes_;
  std::vector<int> indexes_;
};

/*!
 * \brief Factory method for constructing a set from unsorted and non-unique indexes.
 *
 * The indexes in the list a assumed unsorted and could have repeated entries,
 * the effective number of multi-indexes could be less.
 *
 * The size of indexes should be a multiple of the number of dimensions.
 */
indexset make_index_set(int num_dimensions, std::vector<int> const &indexes);

class reindex_map
{
public:
  reindex_map() : num_dimensions_(0), num_active_(0) {}

  reindex_map(int num_dimensions, int num_active) : num_dimensions_(num_dimensions), num_active_(num_active){}

  /*!
   * \brief Sorts the given indexes into an index set and stores a map
   *
   * Holds a map of the sorted and unsorted indexes.
   * The vector holding the indexes is assumed to contain only unique
   * multi-indexes, but sorted in so particular order.
   */
  indexset remap(std::vector<int> const &indexes);

  /*!
   * \brief Map dof to internal order for the kronecker operation.
   */
  template<typename precision>
  void to_ordered(precision const *dof, precision *ordered) const
  {
    for(size_t i=0; i<map_.size(); i++) {
      ordered[i] = (map_[i] < num_active_) ? dof[map_[i]] : precision{0};
      //std::cerr << "ordered[" << i << "] = dof[" << map_[i] << "] = " << dof[map_[i]] << "\n";
    }
  }
  template<typename precision>
  void to_dof(precision const *ordered, precision *dof) const
  {
    //std::cerr << " to dof map \n";
    for(size_t i=0; i<map_.size(); i++)
      if (map_[i] < num_active_) {
        dof[map_[i]] = ordered[i];
        //std::cerr << "dof[" << map_[i] << "] = " << ordered[i] << " i = " << i << "\n";
      }

  }
  template<typename precision>
  void add_to_dof(precision const *ordered, precision *dof) const
  {
    //std::cerr << " to dof map \n";
    for(size_t i=0; i<map_.size(); i++)
      if (map_[i] < num_active_) {
        dof[map_[i]] += ordered[i];
        //std::cerr << "dof[" << map_[i] << "] += " << ordered[i] << " i = " << i << "\n";
      }
  }
  int num_active() const { return num_active_; }

private:
  int num_dimensions_;
  int num_active_;
  std::vector<int> map_;
};

/*!
 * \brief Splits the multi-index set into 1D vectors
 *
 * Using several sort commands (for speed), we identify groups of indexes.
 * For each dimension dim, we have num_vecs(dim) 1d vectors,
 * where the multi-indexes of the 1d vector match in all dimensions but dim.
 * Each of the num_vecs(dim) vectors begins at sorted offset
 * vec_begin(dim, i) and ends at vec_end(dim, i) - 1 (following C++ conventions)
 * where i goes from 0 until num_vecs(dim)-1.
 * The entries of the vector are at global index dimension_sort(dim, j)
 *
 * \code
 *   dimension_sort sorted(iset);
 *   for(int dim=0; dim<iset.num_dimensions(); dim++)
 *     for(int i=0; i<sorted.num_vecs(dim); i++)
 *        for(int j=sorted.vec_begin(dim, i); j<sorted.vec_end(dim, i), j++)
 *          std::cout << " value = " << x[sorted(dim, j)]
 *                    << " at 1d index " << sorted.index1d(dim, j) << "\n";
 * \endcode
 * Note: if the polynomial order is linear or above, each x[] contains
 * (p+1)^num_dimensions entries.
 */
class dimension_sort
{
public:
  //! \brief Empty sort, used for an empty matrix.
  dimension_sort() {}
  //! \brief Sort the indexes dimension by dimension.
  dimension_sort(indexset const &iset);

  //! \brief Number of 1d vectors in dimensions dim
  int num_vecs(int dimension) const { return static_cast<int>(pntr_[dimension].size() -1); }
  //! \brief Begin offset of the i-th vector
  int vec_begin(int dimension, int i) const { return pntr_[dimension][i]; }
  //! \brief End offset (one past the last entry) of the i-th vector
  int vec_end(int dimension, int i) const { return pntr_[dimension][i+1]; }

  //! \brief Get the j-th global offset
  int map(int dimension, int j) const { return map_[dimension][j]; }
  //! \brief Get the 1d index of the j-th entry
  int operator() (indexset const &iset, int dimension, int j) const { return iset[map_[dimension][j]][dimension]; }

private:
  std::vector<std::vector<int>> map_;
  std::vector<std::vector<int>> pntr_;
};

/*!
 * \brief Finds the set of all missing ancestors.
 *
 * Returns the set, so that the union of the given set and the returned set
 * will be ancestry complete.
 *
 * \param iset is the set to be completed
 * \param the lower-triangular part of the connect_1d pattern lists all
 *        ancestors for the ancestors, i.e., row i-th ancestors are in
 *        listed between row_begin(i) and row_diag(i)
 */
indexset compute_ancestry_completion(indexset const &iset,
                                     connect_1d const &pattern1d);


/*!
 * \brief Holds a map between a set of multi-indexes and degrees of freedom.
 */
struct index_map
{
  index_map(int num_dimensions, int num_active, std::vector<int> const &indexes)
    : map(num_dimensions, num_active), iset(map.remap(indexes))
  {}
  reindex_map map;
  indexset iset;
};

/*!
 * \brief Completes the indexes with the edge parents and fills with the basis indexes.
 *
 * \param cell_pattern is used to identify the connections on the level above,
 * it makes it easier to have a single swipe to identify all parents.
 * The \b cell_pattern must be created with connect_1d::level_edge_skip option.
 *
 * \param porder is the polynomial order of the in-cell basis,
 *               e.g., 0 for constants and 2 for quadratics
 */
index_map
complete_and_remap(int num_dimensions, std::vector<int> const &active_cells,
                   connect_1d const &cell_pattern, int porder);

}

