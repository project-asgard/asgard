#pragma once

#include "asgard_function_defs.hpp"

namespace asgard
{

/*!
 * \ingroup asgard_pde_definition
 * \brief Ratio of the circumference to the diameter of a circle
 */
static constexpr double const PI = 3.141592653589793;

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
 * \ingroup asgard_pde_definition
 * \brief Strong-type, usage auto func = separable_func<double>::const_one(number_of_dimensions{3});
 */
enum class number_of_dimensions : int {};

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
    xleft_.fill(0);
    xright_.fill(0);
    length_.fill(0);
  }
  //! create a canonical domain for the given number of dimensions
  pde_domain(int num_dimensions)
    : num_dims_(num_dimensions)
  {
    xleft_.fill(0);
    xright_.fill(1);
    length_.fill(1);
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

    xleft_.fill(0);
    xright_.fill(1);
    length_.fill(1);

    if (list.size() > 0)
      this->set(list);
  }

  //! overwrites the dimension lengths, defaults is (0, 1) in each direction
  void set(std::vector<domain_range> list)
  {
    rassert(list.size() == static_cast<size_t>(num_dims_),
            "provided number of domain_range entries does not match the number of dimensions");

    for (int d : iindexof(num_dims_))
    {
      xleft_[d] = (list.begin() + d)->left;
      xright_[d] = (list.begin() + d)->right;
      length_[d] = xright_[d] - xleft_[d];
      rassert(length_[d] > P{0}, "domain_range specified with negative length");
    }
  }
  //! (for plotting) overwrites the  default names, e.g., x1, x2, x3, v1, v2, v3
  void set_names(std::initializer_list<std::string> list)
  {
    rassert(list.size() == static_cast<size_t>(num_dims_),
            "provided number of names does not match the number of dimensions");

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
  P cell_size(dimension_id dim, int level) const {
    int num_cells = fm::ipow2(level);
    return length_[dim()] / num_cells;
  }
  //! (mostly for moment testing) returns just the position dimensions
  pde_domain<P> position_domain() const {
    if (num_pos_ == 0) {
      if (num_vel_ == 0) return *this; // everything is a position
      else return pde_domain<P>{}; // nothing is a position dimension
    }
    std::vector<domain_range> rng;
    rng.reserve(num_pos_);
    for (int i = 0; i < num_pos_; i++)
      rng.push_back({xleft_[i], xright_[i]});

    pde_domain<P> result(position_dims{num_pos_}, velocity_dims{0});
    result.set(rng);
    return result;
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
 * \brief A function that is the product of 1d functions
 *
 * There are 3 modes of this function, depending on the way that the time
 * component operates. All 3 modes yield identical numerical result; however,
 * the separability and time-invariance can be exploited for better performance,
 * e.g., pre-compute the constant part once and then reuse for each time-step.
 *
 * If the function does not depend on time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...});
 *   // f1, f2, f3 ... have signature asgard::sfixed_func1d<P>
 *   // or f1, f2, f3 ... are just constants
 * \endcode
 *
 * * If the function is separable in time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...}, t);
 *   // f1, f2, f3 ... have signature asgard::sfixed_func1d<P>
 *   // or f1, f2, f3 ... are just constants
 *   // t has signature scalar_func<P>
 * \endcode
 *
 * If the function is non-separable in time:
 * \code
 *   separable_func<P> f({f1, f2, f3, ...});
 *   // f1, f2, f3 ... have signature svector_func1d<P>
 * \endcode
 *
 * The separable function also provides API to query the type and time-dependence in each dimension
 * as well as read the total number of active dimensions.
 * Additionally, each dimension can be reset to a new value with the set() methods.
 *
 */
template<typename P = default_precision>
class separable_func
{
public:
  //! default constructor, no function is set
  separable_func() { funcs_.fill(std::monostate{}); }

  //! set a function that depends on time and is not separable in time
  separable_func(std::vector<svector_func1d<P>> fdomain) : separable_func()
  {
    rassert(static_cast<int>(fdomain.size()) <= max_num_dimensions,
            "separable function provided with too many dimensions, must be up to 6D");
    for (auto i : iindexof(fdomain)) {
      rassert(fdomain[i], "cannot use null function in dimension " + std::to_string(i));
      funcs_[i] = std::move(fdomain[i]);
    }
  }
  //! do not set simultaneously svector_func1d and time function, those can be merged
  separable_func(std::vector<svector_func1d<P>>, scalar_func<P>) : separable_func()
  {
    static_assert(is_valid_call<P>,
        "svector_func1d<P> contains a time variable, so a separate time scalar is not needed. "
        "If the time dependence can be cast as a separable function, it will improve performance. "
        "Otherwise, the time scalar function can be merged within one of the svector_func1d<P>");
  }
  //! set a function that is constant in time
  separable_func(std::vector<sfixed_func1d<P>> fdomain) : separable_func()
  {
    rassert(static_cast<int>(fdomain.size()) <= max_num_dimensions,
            "separable function provided with too many dimensions, must be up to 6D");
    for (auto i : iindexof(fdomain)) {
      rassert(fdomain[i], "cannot use null-function in dimension " + std::to_string(i));
      funcs_[i] = std::move(fdomain[i]);
    }
    time_func_ = P{1};
  }
  //! set a function that is separable in both space and time
  separable_func(std::vector<sfixed_func1d<P>> fdomain, scalar_func<P> f_time)
    : separable_func()
  {
    rassert(static_cast<int>(fdomain.size()) <= max_num_dimensions,
            "separable function provided with too many dimensions, must be up to 6D");
    rassert(f_time, "cannot use null-function for the time dependence");
    for (auto i : iindexof(fdomain)) {
      rassert(fdomain[i], "cannot use null-function in dimension " + std::to_string(i));
      funcs_[i] = std::move(fdomain[i]);
    }
    time_func_ = std::move(f_time);
  }
  //! set a function that is constant throughout the domain but has a time component
  separable_func(std::vector<P> cosnts, scalar_func<P> f_time) : separable_func()
  {
    rassert(static_cast<int>(cosnts.size()) <= max_num_dimensions,
            "separable function provided with too many dimensions, must be up to 6D");
    for (auto i : indexof(cosnts))
      funcs_[i] = std::move(cosnts[i]);
    time_func_ = std::move(f_time);
  }
  //! set a function that is constant throughout space and time
  separable_func(std::vector<P> cosnts) : separable_func()
  {
    rassert(static_cast<int>(cosnts.size()) <= max_num_dimensions,
            "separable function provided with too many dimensions, must be up to 6D");
    for (auto i : indexof(cosnts))
      funcs_[i] = std::move(cosnts[i]);
    time_func_ = P{1};
  }
  //! returns a function that has ones for the given number of dimensions, no time component
  static separable_func<P> const_one(number_of_dimensions num_dims) {
    rassert(static_cast<int>(num_dims) <= max_num_dimensions,
            "too many dimensions, must be up to 6D");
    rassert(static_cast<int>(num_dims) >= 1, "the number of dimensions must be at least 1");
    separable_func<P> result;
    for (int d : iindexof(static_cast<int>(num_dims))) result.funcs_[d] = P{1};
    return result;
  }

  //! check the number of dimensions, does not cache so the cost is not-trivial
  int num_dims() const {
    int dims = 0;
    while (dims < max_num_dimensions and not std::holds_alternative<std::monostate>(funcs_[dims])) ++dims;
    return dims;
  }
  //! returns true if the function is constant in given dimension
  bool is_const(dimension_id dim) const {
    return std::holds_alternative<P>(funcs_[dim()]);
  }
  //! returns true if the function is time-independent in the given dimension
  bool is_fixed(dimension_id dim) const {
    return std::holds_alternative<sfixed_func1d<P>>(funcs_[dim()]);
  }
  //! returns true if the function is (non-separable) time-dependent in the given dimension
  bool is_time_dep(dimension_id dim) const {
    return std::holds_alternative<svector_func1d<P>>(funcs_[dim()]);
  }
  //! returns true if the function is time-dependent and non-separable in time
  bool is_time_non_sep() const {
    return std::holds_alternative<std::monostate>(time_func_);
  }
  //! returns true if the function is time-depend and separable in time
  bool is_time_sep() const {
    return std::holds_alternative<scalar_func<P>>(time_func_);
  }
  //! returns true if the function is constant in time
  bool is_time_const() const {
    return std::holds_alternative<P>(time_func_);
  }
  //! returns the constant function, use only if is_const(dim) is true
  P const_at(dimension_id dim) const { return std::get<P>(funcs_[dim()]); }
  //! returns the fixed function, use only if is_fixed(dim) is true
  sfixed_func1d<P> fixed_at(dimension_id dim) const {
    return std::get<sfixed_func1d<P>>(funcs_[dim()]);
  }
  //! returns the time-dependent function, use only if is_time_dep(dim) is true
  svector_func1d<P> time_dep_at(dimension_id dim) const {
    return std::get<svector_func1d<P>>(funcs_[dim()]);
  }
  //! returns the value of the time-component or 1 if not separable in time
  P time_at(P time) const {
    if (std::holds_alternative<scalar_func<P>>(time_func_))
      return std::get<scalar_func<P>>(time_func_)(time);
    else
      return P{1};
  }
  //! set the given dimension to constant c
  void set(dimension_id dim, P c) {
    rassert(not std::holds_alternative<std::monostate>(funcs_[dim()]),
            "setting constant for invalid dimension " + std::to_string(dim()));
    funcs_[dim()] = c;
  }
  //! set the given dimension to function f
  void set(dimension_id dim, sfixed_func1d<P> f) {
    rassert(not std::holds_alternative<std::monostate>(funcs_[dim()]),
            "setting fixed function for invalid dimension " + std::to_string(dim()));
    funcs_[dim()] = std::move(f);
  }
  //! set the given dimension to function f
  void set(dimension_id dim, svector_func1d<P> f) {
    rassert(not std::holds_alternative<std::monostate>(funcs_[dim()]),
            "setting time-dependant function for invalid dimension " + std::to_string(dim()));
    funcs_[dim()] = std::move(f);
  }
  //! set the time component to be a constant 1
  void set_time_constant() {
    time_func_ = P{1};
  }
  //! set the time component to ft
  void set_time(scalar_func<P> ft) {
    rassert(ft, "invalid time-function");
    time_func_ = std::move(ft);
  }
  //! converts the function to separable in space and non-separable in time
  void set_time_non_separable() {
    time_func_ = std::monostate{};
  }

  //! return true if the current state if invalid, prints to cerr if not consistent
  bool is_valid() const {
    int const nd = num_dims();
    if (is_time_non_sep()) {
      bool any = false;
      for (int d : iindexof(nd)) if (is_time_dep(dimension_id{d})) any = true;
      if (not any) {
        std::cerr << "Found asgard::separable_func set as non-separable in time "
                     "but without a time-depend spacial component for any dimension. "
                     "This 'smells' of an error, so stopping here.\n";
        return false;
      }
    } else {
      bool any = false;
      for (int d : iindexof(nd)) if (is_time_dep(dimension_id{d})) any = true;
      if (any) {
        std::cerr << "Found asgard::separable_func set as separable or constant in time "
                     "but some spacial component have a time-dependence. "
                     "This 'smells' of an error, so stopping here.\n";
        return false;
      }
    }
    for (int d = nd; d < max_num_dimensions; d++) {
      if (not std::holds_alternative<std::monostate>(funcs_[d])) {
        std::cerr << "Found asgard::separable_func where spatial components are set with "
                      "a gap in the dimensions. This should never happen?!\n";
        return false;
      }
    }
    return true;
  }
  //! represents the different ways the time dependence is handled
  enum class time_mode : int { // sync with time_entry
    //! non-separable in time
    non_separable = 0,
    //! constant in time
    constant = 1,
    //! separable in time
    separable = 2
  };

  //! Indicates the time-mode of the function
  time_mode get_time_mode() const { return static_cast<time_mode>(time_func_.index()); }

  //! (testing purposes) eval the function at the points x[] and time t
  P eval(P const x[], P t) {
    std::vector<P> xx(1), fx(1);
    P v = P{1};
    for (int d : iindexof(max_num_dimensions)) {
      v *= std::visit([&](auto const &f) -> P {
          using current_type = std::decay_t<decltype(f)>;
          if constexpr (std::is_same_v<current_type, std::monostate>) {
            return 1; // ignore this dimension
          } else if constexpr (std::is_same_v<current_type, P>) {
            return f;
          } else {
            xx.front() = x[d];
            if constexpr (std::is_same_v<current_type, sfixed_func1d<P>>) {
              f(xx, fx);
              return fx.front();
            } else {
              f(xx, t, fx);
              return fx.front();
            }
          }
        }, funcs_[d]);
    }
    if (is_time_sep()) v *= std::get<scalar_func<P>>(time_func_)(t);
    return v;
  }

  //! writes out general meta-data for the separable function
  void print_stats(std::ostream &os = std::cout) const {
    int nd = num_dims();
    os << "separable function: " << nd << "D\n";
    os << "  (";
    for (int d : iindexof(nd)) {
      if (is_const(dimension_id{d}))
        os << const_at(dimension_id{d});
      else if (is_fixed(dimension_id{d}))
        os << "f" << d << "(x)";
      else
        os << "f" << d << "(x, t)";
      if (d + 1 < nd) os << ", ";
    }
    os << ")";
    if (is_time_const())
      os << " * 1";
    else if (is_time_sep())
      os << " * time(t)";
    os << '\n';
  }
 /*!
  * \ingroup asgard_discretization
  * \brief Allows writing the separable function stats
  */
  friend std::ostream &operator<<(std::ostream &os, separable_func<P> const &func) {
    func.print_stats(os);
    return os;
  }

private:
  // meaning of the type:  monostate -> dimension not set; P -> constant;
  // sfixed -> no-time dep in this dimension; svector_func1d<P> -> non-separable in time
  using func_entry = std::variant<std::monostate, P, sfixed_func1d<P>, svector_func1d<P>>;
  // monostate -> non-separable in time (using svector_func1d); P constant in time (implicit 1);
  // scalar_func<P> -> separable in time
  using time_entry = std::variant<std::monostate, P, scalar_func<P>>;

  std::array<func_entry, max_num_dimensions> funcs_;
  time_entry time_func_ = std::monostate{};
};

/*!
 * \ingroup asgard_discretization
 * \brief Extra data-entry for plotting and post-processing
 *
 * In plotting and post-processing, it is sometime desirable to store
 * additional data that sits on the sparse grid mesh, e.g.,
 * deviation from a nominal state, moments or initial condition.
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
  //! the field can potentially have a different number of dimensions
  int num_dims = -1;
  //! vector data
  std::vector<P> data;
  //! multi-indexes
  std::vector<int> grid;
};

} // namespace asgard
