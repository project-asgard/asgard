#pragma once

#include "asgard_fast_math.hpp"
#include "asgard_matlab_utilities.hpp"
#include "asgard_program_options.hpp"

namespace asgard
{
#ifndef __ASGARD_DOXYGEN_SKIP

template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const, P const)>;

//! vector function using std::vector signature, computes fx(x, t) in 1d
template<typename P>
using svector_func1d = std::function<void(std::vector<P> const &x, P t, std::vector<P> &fx)>;

//! vector function using std::vector signature, computes fx(x) in 1d
template<typename P>
using sfixed_func1d = std::function<void(std::vector<P> const &x, std::vector<P> &fx)>;

//! vector function using std::vector signature and a field, computes fx(x) in 1d
template<typename P>
using sfixed_func1d_f = std::function<void(std::vector<P> const &x, std::vector<P> const &f,
                                           std::vector<P> &fx)>;

template<typename P>
using md_func_type = std::vector<vector_func<P>>;

// same pi used by matlab
static constexpr double const PI = 3.141592653589793;

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using scalar_func = std::function<P(P const)>;

// signature g_func(x, t), may ignore time
template<typename P>
using g_func_type = std::function<P(P const, P const)>;

// uses field-feedback, e.g., g_func_f(x, t, E_field_at_x_t)
template<typename P>
using g_func_f_type = std::function<P(P const, P const, P const)>;


template<typename P>
struct dimension
{
  P domain_min;
  P domain_max;
  std::vector<vector_func<P>> initial_condition;
  g_func_type<P> volume_jacobian_dV;
  std::string name;
  dimension(P const d_min, P const d_max, int const level, int const degree,
            vector_func<P> const initial_condition_in,
            g_func_type<P> const volume_jacobian_dV_in,
            std::string const name_in)

      : dimension(d_min, d_max, level, degree,
                  std::vector<vector_func<P>>({initial_condition_in}),
                  volume_jacobian_dV_in, name_in)
  {}

  dimension(P const d_min, P const d_max, int const level, int const degree,
            std::vector<vector_func<P>> const initial_condition_in,
            g_func_type<P> const volume_jacobian_dV_in,
            std::string const name_in)

      : domain_min(d_min), domain_max(d_max),
        initial_condition(std::move(initial_condition_in)),
        volume_jacobian_dV(volume_jacobian_dV_in), name(name_in)
  {
    expect(domain_min < domain_max);
    set_level(level);
    set_degree(degree);
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }

  void set_level(int const level)
  {
    expect(level >= 0);
    level_ = level;
  }

  void set_degree(int const degree)
  {
    expect(degree >= 0);
    degree_ = degree;
  }

  int level_;
  int degree_;
};

//! usage, pde_domain<double> domain(position_dims{3}, velocity_dims{3});
struct position_dims {
  position_dims() = delete;
  explicit position_dims(int n) : num(n) {}
  int const num;
};
//! usage, pde_domain<double> domain(position_dims{3}, velocity_dims{3});
struct velocity_dims {
  velocity_dims() = delete;
  explicit velocity_dims(int n) : num(n) {}
  int const num;
};

#endif

/*!
 * \ingroup asgard_pde_definition
 * \brief Indicates the left/right end-points of a dimension
 */
template<typename P = default_precision>
struct domain_range {
  //! left end-point
  P left;
  //!  right end-point
  P right;
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
 */
template<typename P = default_precision>
class pde_domain
{
public:
  //! create an empty domain
  pde_domain() {}
  //! create a canonical domain for the given number of dimensions
  pde_domain(int num_dimensions)
    : num_dims_(num_dimensions)
  {
    check_init();
  }
  //! create a domain with given range in each dimension
  pde_domain(std::vector<domain_range<P>> list)
    : num_dims_(static_cast<int>(list.size()))
  {
    check_init();
    this->set(list);
  }
  //! create a canonical domain for the given number of dimensions
  pde_domain(position_dims pos, velocity_dims vel,
             std::initializer_list<domain_range<P>> list = {})
    : num_dims_(pos.num + vel.num), num_pos_(pos.num), num_vel_(vel.num)
  {
    check_init();

    if (list.size() > 0)
      this->set(list);
  }

  //! defaults is (0, 1) in each direction, should probably be overwritten here
  void set(std::initializer_list<domain_range<P>> list)
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
  //! defaults is (0, 1) in each direction, should probably be overwritten here
  void set(std::vector<domain_range<P>> list)
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
  //! (for plotting) default names are x1, x2, x3, v1, v2, v3, can be overwritten
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
  void check_init() {
    if (num_dims_ < 1)
      throw std::runtime_error("pde_domain created with zero or negative dimensions");
    if (num_dims_ > max_num_dimensions)
      throw std::runtime_error("pde_domain created with too many dimensions, max is 6D");
    if (num_pos_ < 0)
      throw std::runtime_error("pde_domain created with negative position dimensions");
    if (num_vel_ < 0)
      throw std::runtime_error("pde_domain created with negative velocity dimensions");

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
  std::array<P, max_num_dimensions> length_ = {{P{1}}};
  std::array<P, max_num_dimensions> xleft_ = {{P{0}}};
  std::array<P, max_num_dimensions> xright_ = {{P{1}}};

  std::array<std::string, max_num_dimensions> dnames_;
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
 *   separable_func<P> f({f1, f2, f3, ...}, separable_func<P>::set_ignore_time);
 * \endcode
 *
 * If a time-independent function is not marked with "set_ignore_time" or if
 * a separable time-component is built into the spacial components, the projection
 * of the function will be recomputed several times per-time step.
 * The result will be the same but there will be some performance penalty.
 */
template<typename P = default_precision>
class separable_func
{
public:
  //! type-tag that indicates the function does not depend on time
  struct type_tag_ignore_time{};
  //! easy way to set the ignore time
  static constexpr type_tag_ignore_time set_ignore_time = type_tag_ignore_time{};

  //! default constructor, no function is set, equivalent to constant 0
  separable_func() = default;

  //! set a function non-separable in time or not depending on time
  separable_func(std::vector<svector_func1d<P>> fdomain)
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    int dims = 0;
    for (auto ip = fdomain.begin(); ip < fdomain.end(); ip++) {
      consts_[dims]        = P{1};
      source_func_[dims++] = std::move(*ip);
    }
  }
  //! set a function non-separable in time or not depending on time
  separable_func(std::vector<svector_func1d<P>> fdomain, type_tag_ignore_time)
    : ignores_time_(true)
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    int dims = 0;
    for (auto ip = fdomain.begin(); ip < fdomain.end(); ip++) {
      consts_[dims]        = P{1};
      source_func_[dims++] = std::move(*ip);
    }
  }
  //! set a function that is separable in both space and time
  separable_func(std::vector<svector_func1d<P>> fdomain, scalar_func<P> f_time)
    : time_func_(std::move(f_time))
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);

    int dims = 0;
    for (auto ip = fdomain.begin(); ip < fdomain.end(); ip++) {
      consts_[dims]        = P{1};
      source_func_[dims++] = std::move(*ip);
    }
  }
  //! set a function that is constant throughout the domain but has a time component
  separable_func(std::vector<P> fdomain, scalar_func<P> f_time)
    : time_func_(std::move(f_time))
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    std::copy(fdomain.begin(), fdomain.end(), consts_.begin());
  }
  //! set a function that is constant throughout the domain but has a time component
  separable_func(std::vector<P> fdomain)
    : ignores_time_(true)
  {
    expect(static_cast<int>(fdomain.size()) <= max_num_dimensions);
    std::copy(fdomain.begin(), fdomain.end(), consts_.begin());
  }

  //! check the number of dimensions, does not cache use primarily for verification
  int num_dims() const {
    int dims = 0;
    for (auto const &s : consts_) if (s != 0) dims++;
    return dims;
  }

  //! returns the i-th domain function
  svector_func1d<P> const &fdomain(int i) const { return source_func_[i]; }
  //! returns the i-th constant function
  P cdomain(int i) const { return consts_[i]; }
  //! set the i-th function to f
  void set_fdomain(int i, svector_func1d<P> f) {
    source_func_[i] = std::move(f);
    consts_[i] = 1;
  }
  //! sets the i-th function to a constant function
  void set_cdomain(int i, P c) {
    expect(c != 0);
    source_func_[i] = nullptr;
    consts_[i] = c;
  }
  //! applies the i-th domain function on x and return the result in y
  void fdomain(int i, std::vector<P> const &x, P t, std::vector<P> &y) const {
    return source_func_[i](x, t, y);
  }
  //! check if the given dimension is constant
  bool is_const(int dim) const { return (not source_func_[dim]); }

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
      if (source_func_[d]) {
        xx.front() = x[d];
        source_func_[d](xx, t, fx);
        v *= fx[0];
      } else if (consts_[d] != 0) {
        v *= consts_[d];
      }
    }
    if (time_func_)
      v *= time_func_(t);
    return v;
  }

private:
  bool ignores_time_ = false;
  std::array<svector_func1d<P>, max_num_dimensions> source_func_;
  std::array<P, max_num_dimensions> consts_ = {{0}};
  scalar_func<P> time_func_;
};

} // namespace asgard
