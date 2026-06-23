#pragma once

#include "asgard_indexset.hpp"

namespace asgard
{

/*!
 * \ingroup asgard_pde_definition
 * \brief Strong-type, usage separable_func<double> func; func.set(dimension_id{2}, val);
 */
struct dimension_id {
  //! do not create an empty dimension id
  dimension_id() = delete;
  //! set the index dimension
  explicit dimension_id(int n) : id(n) {
    rassert(0 <= n and n < max_num_dimensions,
            "invalid dimension, must be in 0 ... 5 for 1D through 6D problems");
  }
  //! holds the id of the position dimensions
  int const id;
  //! returns the index with a simple call
  int operator()() const { return id; }
};

/*!
 * \ingroup asgard_funcdef
 * \brief Holds the power coefficients for the moments
 */
struct moment
{
  //! produces zero moment with the specified number of velocity dimensions
  static moment zero(int num_velocity) {
    moment m(0);
    for (int d = 1; d < num_velocity; d++) m.pows[d] = 0;
    return m;
  }
  //! produces a moment for the electric field in the dim dimension
  //! i.e. dim = dimension_id(0) corresponds to E_x
  static moment electric(dimension_id dim, int num_pos_dims) {
    moment m = moment::zero(num_pos_dims);
    m.pows[dim()] = electric_flag;
    return m;
  }
  //! creating a placeholder (invalid) moment
  moment() : pows{unset_flag, unset_flag, unset_flag} {}
  //! create a 1D moment with the given power
  moment(int pv1) : pows{pv1, unset_flag, unset_flag} {}
  //! create a 2D moment with the given powers
  moment(int pv1, int pv2) : pows{pv1, pv2, unset_flag} {}
  //! create a 3D moment with the given powers
  moment(int pv1, int pv2, int pv3) : pows{pv1, pv2, pv3} {}
  //! number of valid powers
  int num_dims() const {
    for (int i = 0; i < max_mom_dims; i++)
      if (pows[i] == unset_flag) return i;
    return max_mom_dims;
  }
  //! return the d-th power
  int operator [] (int d) const { return pows[d]; }
  //! check whether two moments are the same
  bool operator == (moment const &other) const {
    return std::equal(pows.begin(), pows.end(), other.pows.begin());
  }
  //! check whether two moments are different
  bool operator != (moment const &other) const {
    return not (*this == other);
  }
  //! check whether a moment is an electric field moment
  bool is_electric() const {
    for (int const pow : pows)
      if (pow == electric_flag) return true;
    return false;
  }
  //! get the direction of the electric field corresponding to the moment
  int get_electric_direction() const {
    for (int const i : iindexof(pows))
      if (pows[i] == electric_flag) return i;
    rassert(false, "The moment must be an electric field moment to get its direction");
    return -1; // unreachable due to rassert
  }
  //! convert the moment to a string containing the powers (consistent with python)
  std::string to_string() const {
    std::string m = (pows[0] == unset_flag) ? "x" : std::to_string(pows[0]);
    m += (pows[1] == unset_flag) ? "x" : std::to_string(pows[1]);
    m += (pows[2] == unset_flag) ? "x" : std::to_string(pows[2]);
    return m;
  }
  //! print information about the moment to an std::ostream
  friend std::ostream& operator<<(std::ostream& os, moment const &m) {
    if (m.num_dims() == 0) os << "()";
    else if (m.num_dims() == 1) os << "(" << m.pows[0] << ")";
    else {
      os << "(" << m.pows[0];
      for (int i = 1; i < m.num_dims(); i++)
        os << ", " << m.pows[i];
      os << ")";
    }
    return os;
  }

  //! holds the powers
  std::array<int, max_mom_dims> pows;

  private:
    static int const unset_flag = -1;
    static int const electric_flag = -0xef; // ef for electric field, this has a decimal value of -239
};

/*!
 * \ingroup asgard_funcdef
 * \brief Strong type for the moment ID
 *
 * Wrapper around an int that can be used to access a specific moment from asgard::momentset
 * and asgard::momentset_gpu
 *
 * The moment-id is obtained by calling pde_scheme::register_moment and should not be created
 * directly from an int, let ASGarD do the correct initialization.
 */
class moment_id {
public:
  //! default placeholder id
  moment_id() = default;
  //! explicit constructor for the new id
  explicit constexpr moment_id(int num) : id_(num) {}
  //! get the id
  constexpr int get() const { return id_; }
  //! another getter
  constexpr int operator () () const { return id_; }

  //! check whether two ids are the same
  constexpr bool operator == (moment_id const &other) const {
    return id_ == other.id_;
  }
  //! check whether two ids are different
  constexpr bool operator != (moment_id const &other) const {
    return not (*this == other);
  }
  //! unset moment, cannot be used inside asgard::momentset
  static constexpr moment_id unset() { return moment_id{-1}; }

private:
  //! stored value for the ID
  int id_ = unset()();
};

#ifndef __ASGARD_DOXYGEN_SKIP

/*!
 * \brief Holds the list of moments and manages the ids
 */
class moments_list {
public:
  //! creates a new empty list
  moments_list();
  //! returns the current number of moments
  int num_moms() const { return static_cast<int>(moms_.size()); }
  //! returns the current number of moments
  int size() const { return static_cast<int>(moms_.size()); }
  //! returns true if the list is empty
  bool empty() const { return moms_.empty(); }

  //! \brief adds a new moment to the set
  void add_moment(moment const &mom) {
    this->get_add_id(mom);
  }
  //! \brief returns the ID of the moment, adds the moment to the list (if not there already)
  moment_id get_add_id(moment const &mom) {
    for (int const i : iindexof(moms_))
      if (moms_[i] == mom)
        return moment_id{i};
    moms_.push_back(mom);
    return moment_id{static_cast<int>(moms_.size() - 1)};
  }
  //! returns the ID of the moment if it exists, otherwise it returns an unset moment ID
  moment_id get_check_id(moment const &mom) const {
    for (int const i : iindexof(moms_))
      if (moms_[i] == mom)
        return moment_id{i};
    return moment_id::unset();
  }
  //! returns the ID of an already existing moment
  moment_id get_id(moment const &mom) const {
    for (int const i : iindexof(moms_))
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
  //! true if a moment requires a poisson solver
  bool has_electric() {
    for(moment const &mom : moms_)
      if (mom.is_electric())
        return true;
    return false;
  }

private:
  std::vector<moment> moms_;
};

#endif

/*!
 * \ingroup asgard_funcdef
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

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    size_t t = 0;
    for (auto const &v : moms_) t += v.size();
    return t * sizeof(P);
  }

private:
  std::vector<std::vector<P>> moms_;
};

#ifdef ASGARD_USE_GPU
/*!
 * \ingroup asgard_funcdef
 * \brief Holds the computed moments on the GPU
 *
 * Stores the data for each moment after it has been computed,
 * can hold either the hierarchical coefficients or the interpolation values.
 *
 * The returned gpu::vector objects work similar to std::vector, they have a .data() method
 * that returns a raw-pointer to the data and .size() that returns int64_t value.
 * Individual entries cannot be dereferenced from the CPU.
 */
template<typename P>
class momentset_gpu {
public:
  //! create an empty moment list
  momentset_gpu() = default;
  //! create the new set with the given number of moments
  momentset_gpu(int num_moments) : moms_(num_moments) {}

  //! returns the number of stored moments
  size_t size() const { return moms_.size(); }

  //! return the provided moment, const variant
  gpu::vector<P> const &operator[] (moment_id mid) const { return moms_[mid()]; }
  //! return the provided moment
  gpu::vector<P> &operator[] (moment_id mid) { return moms_[mid()]; }
  //! return the provided moment, never const
  gpu::vector<P> &get(moment_id mid) { return moms_[mid()]; }
  //! return the raw-array for the provided moment
  P const *data(moment_id mid) const { return moms_[mid()].data(); }

  //! computes approximate memory usage by the object
  size_t used_bytes() const {
    size_t t = 0;
    for (auto const &v : moms_) t += v.size();
    return t * sizeof(P);
  }

private:
  std::vector<gpu::vector<P>> moms_;
};
#else
// placeholder type, cannot be used without enabled GPU
template<typename P>
class momentset_gpu {};
#endif

}
