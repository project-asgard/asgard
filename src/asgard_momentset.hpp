#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{

/*!
 * \brief Maximum number of dimensions for the moments
 *
 * Do not change unless you really know what you're doing.
 */
inline constexpr int max_mom_dims = 3;

/*!
 * \brief Holds the coefficients for the moments
 */
struct moment
{
  //! create a 1D moment with the given power
  moment(int pv1) : pows{pv1, -1, -1} {}
  //! create a 2D moment with the given powers
  moment(int pv1, int pv2) : pows{pv1, pv2, -1} {}
  //! create a 3D moment with the given powers
  moment(int pv1, int pv2, int pv3) : pows{pv1, pv2, pv3} {}
  //! number of valid powers
  int num_dims() const {
    for (int i = 0; i < max_mom_dims; i++)
      if (pows[i] < 0) return i;
    return max_mom_dims;
  }
  //! check whether two moments are the same
  bool operator == (moment const &other) const {
    return std::equal(pows.begin(), pows.end(), other.pows.begin());
  }
  //! check whether two moments are different
  bool operator != (moment const &other) const {
    return not (*this == other);
  }
  //! holds the powers
  std::array<int, max_mom_dims> pows;
};

std::ostream& operator<<(std::ostream& os, moment const &m);

//! strong type for the moment ID
class moment_id {
public:
  //! explicit constructor for the new id
  explicit moment_id(int num) : id_(num) {}
  //! get the id
  int get() const { return id_; }
  //! another getter
  int operator () () const { return id_; }

  //! check whether two ids are the same
  bool operator == (moment_id const &other) const {
    return id_ == other.id_;
  }
  //! check whether two ids are different
  bool operator != (moment_id const &other) const {
    return not (*this == other);
  }

private:
  //! stored value for the ID
  int id_;
};

/*!
 * \brief Holds the list of moments and manages the ids
 */
class moments_list {
public:
  //! creates a new empty list
  moments_list() {
    moms_.reserve(7); // should rarely be exceeded
  }
  //! returns the current number of moments
  int num_moms() const { return static_cast<int>(moms_.size()); }
  //! returns the current number of moments
  int size() const { return static_cast<int>(moms_.size()); }
  //! returns true if the list is empty
  bool empty() const { return moms_.empty(); }

  //! \brief adds a new moment to the set
  void add_moment(moment const &mom) {
    this->get_id(mom);
  }
  //! \brief returns the ID of the moment, adds the moment to the list (if not there already)
  moment_id get_id(moment const &mom) {
    for (int i = 0; i < static_cast<int>(moms_.size()); i++)
      if (moms_[i] == mom)
        return moment_id{i};
    moms_.push_back(mom);
    return moment_id{static_cast<int>(moms_.size() - 1)};
  }
  //! returns the ID of an already existing moment
  moment_id get_id(moment const &mom) const {
    for (int i = 0; i < static_cast<int>(moms_.size()); i++)
      if (moms_[i] == mom)
        return moment_id{i};
    throw std::runtime_error("cannot find the specified moment");
  }
  //! return the moment corresponding to the given ID
  moment const &operator[] (moment_id mid) const { return moms_[mid()]; }
  //! return the moment with the given index
  moment const &operator[] (int i) const { return moms_[i]; }

  //! returns true if all moments have the given dimension
  bool have_all_dimension(int const dims) const;
  //! returns moment_id of the members of this list within the main set
   std::vector<moment_id> find_as_subset_of(moments_list const &superset) const;

  //! returns the max powers in each dimension
  moment max_moment() const;
  //! returns the max powers in specific dimension
  int max_moment(int dim) const;
  //! print the list
  void print(std::ostream &os = std::cout) const {
    for (auto const &m : moms_)
      os << m << "  ";
  }

private:
  std::vector<moment> moms_;
};

/*!
 * \brief Holds the computed moments
 *
 * Stores the data for each moment after it has been computed,
 * can hold either the hierarchical coefficients or the interpolation values.
 */
template<typename P>
class momentset {
public:
  //! create an empty moment list
  momentset() = default;
  //! create the new set with the given number of moments
  momentset(int num_moments) : moms_(num_moments) {}

  //! returns the number of stored moments
  size_t size() const { return moms_.size(); }

  //! return the provided moment, const variant
  std::vector<P> const &operator[] (moment_id mid) const { return moms_[mid()]; }
  //! return the provided moment
  std::vector<P> &operator[] (moment_id mid) { return moms_[mid()]; }
  //! return the provided moment, never const
  std::vector<P> &get(moment_id mid) { return moms_[mid()]; }

private:
  std::vector<std::vector<P>> moms_;
};

}
