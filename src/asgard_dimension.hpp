#pragma once

#include "asgard_resource_groups.hpp"
#include "asgard_program_options.hpp"

namespace asgard
{

/*!
 * \ingroup asgard_pde_definition
 * \brief Vector function, computing fx = F(t, x)
 */
template<typename P>
using svector_func1d = std::function<void(std::vector<P> const &x, P t, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_pde_definition
 * \brief Vector function, computing fx = F(x), no time-dependence
 */
template<typename P>
using sfixed_func1d = std::function<void(std::vector<P> const &x, std::vector<P> &fx)>;

/*!
 * \ingroup asgard_pde_definition
 * \brief Vector function, computing fx = F(x, f), where f is a field, e.g., moment of the solution
 */
template<typename P>
using sfixed_func1d_f = std::function<void(std::vector<P> const &x, std::vector<P> const &f,
                                           std::vector<P> &fx)>;

/*!
 * \ingroup asgard_pde_definition
 * \brief Ratio of the circumference to the diameter of a circle
 */
static constexpr double const PI = 3.141592653589793;

/*!
 * \ingroup asgard_pde_definition
 * \brief Scalar function, returning f(x)
 */
template<typename P>
using scalar_func = std::function<P(P const)>;

/*!
 * \ingroup asgard_pde_definition
 * \brief Strong-type, usage: pde_domain<double> domain(position_dims{3}, velocity_dims{3});
 */
struct position_dims {
  //! do not create an empty position dimension number
  position_dims() = delete;
  //! sets the position dimensions
  explicit position_dims(int n) : num(n) {}
  //! holds the number of position dimensions
  int const num;
};
/*!
 * \ingroup asgard_pde_definition
 * \brief Strong-type, usage: pde_domain<double> domain(position_dims{3}, velocity_dims{3});
 */
struct velocity_dims {
  //! do not create an empty velocity dimension number
  velocity_dims() = delete;
  //! sets the velocity dimensions
  explicit velocity_dims(int n) : num(n) {}
  //! holds the number of position dimensions
  int const num;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Indicates the left/right end-points of a dimension
 */
struct domain_range {
  //! make a range from l to r, i.e., [l, r]
  domain_range(double l, double r) : left(l), right(r) {}
  //! left end-point
  double left;
  //!  right end-point
  double right;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Defines a domain for the PDE
 *
 * First we specify the number of dimensions, could be a single number
 * or split between position and velocity.
 * The split allows for better management of kinetic problems,
 * such as computing moments and using builtin operators that depend
 * on the moments.
 * If such operators are not used, then the split is meaningless.
 *
 * Second, we specify the side of the domain in each direction, the domain
 * is a multidimensional hyper-box.
 */
template<typename P = default_precision>
class pde_domain
{
public:
  //! create an empty domain
  pde_domain() {
    std::fill(xleft_.begin(), xleft_.end(), 0);
    std::fill(xright_.begin(), xright_.end(), 0);
    std::fill(length_.begin(), length_.end(), 0);
  }
  //! create a canonical domain for the given number of dimensions
  pde_domain(int num_dimensions)
    : num_dims_(num_dimensions)
  {
    std::fill(xleft_.begin(), xleft_.end(), 0);
    std::fill(xright_.begin(), xright_.end(), 1);
    std::fill(length_.begin(), length_.end(), 1);
    check_init();
  }
  //! create a domain with given range in each dimension
  pde_domain(std::vector<domain_range> list)
    : num_dims_(static_cast<int>(list.size()))
  {
    check_init();
    this->set(list);
  }
  //! create a canonical domain for the given number of dimensions
  pde_domain(position_dims pos, velocity_dims vel,
             std::vector<domain_range> list = {})
    : num_dims_(pos.num + vel.num), num_pos_(pos.num), num_vel_(vel.num)
  {
    check_init();

    std::fill(xleft_.begin(), xleft_.end(), 0);
    std::fill(xright_.begin(), xright_.end(), 1);
    std::fill(length_.begin(), length_.end(), 1);

    if (list.size() > 0)
      this->set(list);
  }

  //! overwrites the dimension lengths, defaults is (0, 1) in each direction
  void set(std::initializer_list<domain_range> list)
  {
    if (static_cast<int>(list.size()) != num_dims_)
      throw std::runtime_error("provided number of domain_range entries does not match the "
                               "number of dimensions");

    for (int d : iindexof(num_dims_))
    {
      xleft_[d] = (list.begin() + d)->left;
      xright_[d] = (list.begin() + d)->right;
      length_[d] = xright_[d] - xleft_[d];
      if (length_[d] < P{0})
        throw std::runtime_error("domain_range specified with negative length");
    }
  }
  //! overwrites the dimension lengths, defaults is (0, 1) in each direction
  void set(std::vector<domain_range> list)
  {
    if (static_cast<int>(list.size()) != num_dims_)
      throw std::runtime_error("provided number of domain_range entries does not match the "
                               "number of dimensions");

    for (int d : iindexof(num_dims_))
    {
      xleft_[d] = (list.begin() + d)->left;
      xright_[d] = (list.begin() + d)->right;
      length_[d] = xright_[d] - xleft_[d];
      if (length_[d] < P{0})
        throw std::runtime_error("domain_range specified with negative length");
    }
  }
  //! (for plotting) overwrites the  default names, e.g., x1, x2, x3, v1, v2, v3
  void set_names(std::initializer_list<std::string> list)
  {
    if (static_cast<int>(list.size()) != num_dims_)
      throw std::runtime_error("provided number of names does not match the "
                               "number of dimensions");

    for (int d : iindexof(num_dims_))
      dnames_[d] = *(list.begin() + d);
  }

  //! returns the number of dimension
  int num_dims() const { return num_dims_; }
  //! returns the number of position dimensions (if set)
  int num_pos() const { return num_pos_; }
  //! returns the number of velocity dimensions (if set)
  int num_vel() const { return num_vel_; }

  //! returns the length in dimension d
  P length(int d) const { return length_[d]; }
  //! returns the left point of dimension d
  P xleft(int d) const { return xleft_[d]; }
  //! returns the right point of dimension d
  P xright(int d) const { return xright_[d]; }

  //! returns the name of dimension d
  std::string const &name(int i) { return dnames_[i]; }

  //! (related to cfl) given the provided maximum level, find the smallest cell size
  P min_cell_size(int max_level) const {
    int num_cells = fm::ipow2(max_level);
    P msize = length_[0] / num_cells;
    for (int d = 1; d < num_dims_; d++)
      msize = std::min(msize, length_[d] / num_cells);
    return msize;
  }
  //! returns the cell-size for given dimension and level, uses the length
  P cell_size(int dim, int level) const {
    int num_cells = fm::ipow2(level);
    return length_[dim] / num_cells;
  }

  //! used for i/o purposes
  friend class h5manager<P>;

private:
  //! verify the consistency of the provided conditions
  void check_init() {
    rassert(num_pos_ >= 0, "pde_domain created with negative position dimensions");
    rassert(num_vel_ >= 0, "pde_domain created with negative velocity dimensions");
    rassert(num_dims_ >= 1, "pde_domain created with zero or negative dimensions");
    rassert(num_dims_ <= max_num_dimensions,
            "pde_domain created with too many dimensions, max is 6D");

    if (num_pos_ == 0 and num_vel_ == 0) {
      for (int d : iindexof(num_dims_))
        dnames_[d] = "x" + std::to_string(d + 1);
    } else {
      for (int d : iindexof(num_pos_))
        dnames_[d] = "x" + std::to_string(d + 1);
      for (int d : iindexof(num_vel_))
        dnames_[d + num_pos_] = "v" + std::to_string(d + 1);
    }
  }

  int num_dims_ = 0;
  int num_pos_ = 0;
  int num_vel_ = 0;
  std::array<P, max_num_dimensions> length_;
  std::array<P, max_num_dimensions> xleft_;
  std::array<P, max_num_dimensions> xright_;

  std::array<std::string, max_num_dimensions> dnames_;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief Type-tag that indicates that a separable_func function does not depend on time
 */
struct type_tag_ignores_time{};
/*!
 * \ingroup asgard_pde_definition
 * \brief Easy shortcut to indicate that a separable_func ignores time
 */
inline constexpr type_tag_ignores_time ignores_time = type_tag_ignores_time{};

/*!
 * \ingroup asgard_discretization
 * \brief Indicates a separable function with ones in the given number of dimensions
 */
struct ones_for_dimensions {
  //! sets the number of dimensions
  ones_for_dimensions(int d = 0) : dims(d) {
    rassert(0 < dims and dims < max_num_dimensions,
            "incorrect number of dimensions for ones_for_dimensions()");
  }
  //! number of dimensions
  int dims = 0;
};

/*!
 * \ingroup asgard_pde_definition
 * \brief A function that is the product of 1d functions
 *
 * There are 3 modes of this function, depending on the way that the time
 * component operates.
 *
 * If the function is non-separable in time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...});
 *   // f1 has signature svector_func1d<P>
 * \endcode
 *
 * If the function is separable in time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...}, t);
 *   // t has signature scalar_func<P>
 * \endcode
 *
 * If the function does not depend on time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...}, ignores_time);
 * \endcode
 *
 * If a time-independent function is not marked with "ignores_time" or if
 * a separable time-component is built into the spacial components, the projection
 * of the function will be recomputed several times per-time step.
 * The result will be the same but there will be some performance penalty.
 */
template<typename P = default_precision>
class separable_func
{
public:
  //! default constructor, no function is set
  separable_func() = default;

  //! set a function that depends on time and is not separable in time
  separable_func(std::vector<svector_func1d<P>> fdomain)
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    for (auto i : indexof(fdomain))
      funcs_[i] = std::move(fdomain[i]);
  }
  //! set a function that is constant in time
  separable_func(std::vector<svector_func1d<P>> fdomain, type_tag_ignores_time)
    : ignores_time_(true)
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    for (auto i : indexof(fdomain))
      funcs_[i] = std::move(fdomain[i]);
  }
  //! set a function that is separable in both space and time
  separable_func(std::vector<svector_func1d<P>> fdomain, scalar_func<P> f_time)
    : time_func_(std::move(f_time))
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    for (auto i : indexof(fdomain))
      funcs_[i] = std::move(fdomain[i]);
  }
  //! set a function that is constant throughout the domain but has a time component
  separable_func(std::vector<P> cosnts, scalar_func<P> f_time)
    : time_func_(std::move(f_time))
  {
    expect(static_cast<int>(cosnts.size()) <= max_num_dimensions);
    for (auto i : indexof(cosnts))
      funcs_[i] = cosnts[i];
  }
  //! set a function that is constant throughout space and time
  separable_func(std::vector<P> const &cosnts)
    : ignores_time_(true)
  {
    expect(static_cast<int>(cosnts.size()) <= max_num_dimensions);
    for (auto i : indexof(cosnts))
      funcs_[i] = cosnts[i];
  }
  //! sets ones in the given number of dimensions
  separable_func(ones_for_dimensions const &ones) {
    for (auto i : indexof(ones.dims))
      funcs_[i] = 1;
  }
  //! sets ones in the given number of dimensions
  separable_func(ones_for_dimensions const &ones, type_tag_ignores_time)
    : ignores_time_(true)
  {
    for (auto i : indexof(ones.dims))
      funcs_[i] = 1;
  }

  //! check the number of dimensions, does not cache so the cost is not-trivial
  int num_dims() const {
    int dims = 0;
    for (auto const &f : funcs_) if (f.index() != 0) dims++;
    return dims;
  }

  //! returns the i-th domain function
  svector_func1d<P> const &fdomain(int i) const { return std::get<2>(funcs_[i]); }
  //! returns the i-th constant function
  P cdomain(int i) const { return std::get<1>(funcs_[i]); }
  //! set the i-th function to f
  void set(int i, svector_func1d<P> f) {
    funcs_[i] = std::move(f);
  }
  //! sets the i-th function to a constant function
  void set(int i, P c) {
    funcs_[i] = c;
  }
  //! applies the i-th domain function on x and return the result in y
  void fdomain(int i, std::vector<P> const &x, P t, std::vector<P> &y) const {
    return std::get<2>(funcs_[i])(x, t, y);
  }
  //! check if the given dimension is constant
  bool is_const(int dim) const { return (funcs_[dim].index() == 1); }

  //! returns the time function
  scalar_func<P> const &ftime() const { return time_func_; }
  //! returns the value of the time function
  P ftime(P t) const { return time_func_(t); }

  //! returns true if the function is set to ignore times
  bool ignores_time() const { return ignores_time_; }
  //! returns true if the function is separable in time
  bool separable_time() const { return (!!time_func_ or ignores_time_); }

  //! (testing purposes) eval the function at the points x[] and time t
  P eval(P const x[], P t) {
    std::vector<P> xx(1), fx(1);
    P v = P{1};
    for (int d : iindexof(max_num_dimensions)) {
      if (funcs_[d].index() == 2) {
        xx.front() = x[d];
        std::get<2>(funcs_[d])(xx, t, fx);
        v *= fx[0];
      } else if (funcs_[d].index() == 1) {
        v *= std::get<1>(funcs_[d]);
      }
    }
    if (time_func_)
      v *= time_func_(t);
    return v;
  }

private:
  using func_entry = std::variant<int, P, svector_func1d<P>>;

  bool ignores_time_ = false;
  std::array<func_entry, max_num_dimensions> funcs_;
  scalar_func<P> time_func_;
};

/*!
 * \ingroup asgard_discretization
 * \brief Extra data-entry for plotting and post-processing
 *
 * In plotting and post-processing, it is sometime desirable to store
 * additional data that sits on the sparse grid mesh, e.g.,
 * deviation from a nominal state or initial condition.
 * Since the data is defined on a sparse grid, it has to be accessed with
 * the asgard::reconstruct_solution class (e.g., via python), but the data
 * has to be saved/loaded in the asgard::discretization_manager
 */
template<typename P>
struct aux_field_entry {
  //! default constructor, creates and empty entry
  aux_field_entry() = default;
  //! constructor, set the name and data
  aux_field_entry(std::string nm, std::vector<P> dat)
      : name(std::move(nm)), data(std::move(dat))
  {}
  //! reference name for the field, should be unique
  std::string name;
  //! vector data
  std::vector<P> data;
  //! multi-indexes
  std::vector<int> grid;
};

} // namespace asgard
