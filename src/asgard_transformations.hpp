#pragma once
#include "asgard_batch.hpp"

namespace asgard
{
template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    PDE<P> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
fk::vector<P> gen_realspace_nodes(
    int const degree, int const level, P const min, P const max,
    quadrature_mode const quad_mode = quadrature_mode::use_fixed);

template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    std::vector<dimension<P>> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
void wavelet_to_realspace(
    PDE<P> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

template<typename P>
void wavelet_to_realspace(
    std::vector<dimension<P>> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space,
    quadrature_mode const quad_mode = quadrature_mode::use_degree);

// overload - get only the elements of the combined vector that fall within a
// specified range
template<typename P>
fk::vector<P>
combine_dimensions(int const, elements::table const &, int const, int const,
                   std::vector<fk::vector<P>> const &, P const = 1.0);

template<typename P>
void combine_dimensions(int const degree, elements::table const &table,
                        int const start_element, int const stop_element,
                        std::vector<fk::vector<P>> const &vectors,
                        P const time_scale,
                        fk::vector<P, mem_type::view> result);

template<typename P, typename F>
fk::vector<P> forward_transform(
    dimension<P> const &dim, F function, g_func_type<P> dv_func,
    basis::wavelet_transform<P, resource::host> const &transformer,
    P const time = 0)
{
  int const num_levels = dim.get_level();
  int const degree     = dim.get_degree();
  P const domain_min   = dim.domain_min;
  P const domain_max   = dim.domain_max;

  expect(num_levels >= 0);
  expect(num_levels <= transformer.max_level);
  expect(degree >= 0);
  expect(domain_max > domain_min);

  // check to make sure the F function arg is a function type
  // that will accept a vector argument. we have a check for its
  // return below
  static_assert(std::is_invocable_v<decltype(function), fk::vector<P>, P>);

  // get the Legendre-Gauss nodes and weights on the domain
  // [-1,+1] for performing quadrature.
  auto const [roots, weights] =
      legendre_weights<P>(degree, -1, 1, quadrature_mode::use_fixed);

  // get grid spacing.
  // hate this name TODO
  int const n                  = fm::two_raised_to(num_levels);
  int const degrees_freedom_1d = (degree + 1) * n;

  // get the Legendre basis function evaluated at the Legendre-Gauss nodes   //
  // up to order k
  P const normalize         = (domain_max - domain_min) / n;
  fk::matrix<P> const basis = [&roots = roots, degree, normalize] {
    fk::matrix<P> legendre_ = legendre<P>(roots, degree)[0];
    return legendre_.transpose() * (static_cast<P>(1.0) / std::sqrt(normalize));
  }();

  // this will be our return vector
  fk::vector<P> transformed(degrees_freedom_1d);

  // initial condition for f
  // hate this name also TODO

  for (int i = 0; i < n; ++i)
  {
    // map quad_x from [-1,+1] to [domain_min,domain_max] physical domain.
    fk::vector<P> const mapped_roots = [&roots = roots, normalize, domain_min,
                                        i]() {
      fk::vector<P> out(roots.size());
      std::transform(out.begin(), out.end(), roots.begin(), out.begin(),
                     [&](P &elem, P const &root) {
                       return elem + (normalize * (root / 2.0 + 1.0 / 2.0 + i) +
                                      domain_min);
                     });
      return out;
    }();

    // get the f(v) initial condition at the quadrature points.
    fk::vector<P> f_here = function(mapped_roots, time);

    // apply dv to f(v)
    if (dv_func)
    {
      std::transform(f_here.begin(), f_here.end(), mapped_roots.begin(),
                     f_here.begin(),
                     [dv_func, time](P &f_elem, P const &x_elem) -> P {
                       return f_elem * dv_func(x_elem, time);
                     });
    }

    // ensuring function returns vector of appropriate size
    expect(f_here.size() == weights.size());
    std::transform(f_here.begin(), f_here.end(), weights.begin(),
                   f_here.begin(), std::multiplies<P>());

    // generate the coefficients for DG basis
    fk::vector<P> coeffs = basis * f_here;

    transformed.set_subvector(i * (degree + 1), coeffs);
  }
  transformed = transformed * (normalize / 2.0);

  // transfer to multi-DG bases
  transformed =
      transformer.apply(transformed, dim.get_level(), basis::side::left,
                        basis::transpose::no_trans);

  // zero out near-zero values resulting from transform to wavelet space
  std::transform(transformed.begin(), transformed.end(), transformed.begin(),
                 [](P &elem) {
                   return std::abs(elem) < std::numeric_limits<P>::epsilon()
                              ? static_cast<P>(0.0)
                              : elem;
                 });

  return transformed;
}

template<typename P>
fk::vector<P> sum_separable_funcs(
    std::vector<md_func_type<P>> const &funcs,
    std::vector<dimension<P>> const &dims,
    adapt::distributed_grid<P> const &grid,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const degree, P const time);

template<typename P>
inline fk::vector<P> transform_and_combine_dimensions(
    PDE<P> const &pde, std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time = 0.0,
    P const time_multiplier = 1.0)
{
  expect(static_cast<int>(v_functions.size()) == pde.num_dims());
  expect(start <= stop);
  expect(stop < table.size());
  expect(degree >= 0);

  std::vector<fk::vector<P>> dimension_components;
  dimension_components.reserve(pde.num_dims());

  auto const &dimensions = pde.get_dimensions();

  for (int i = 0; i < pde.num_dims(); ++i)
  {
    auto const &dim = dimensions[i];
    dimension_components.push_back(forward_transform<P>(
        dim, v_functions[i], dim.volume_jacobian_dV, transformer, time));
    int const n = dimension_components.back().size();
    std::vector<int> ipiv(n);
    expect(dim.get_mass_matrix().nrows() >= n);
    expect(dim.get_mass_matrix().ncols() >= n);
    fk::matrix<P> lhs_mass =
        dim.get_mass_matrix().extract_submatrix(0, 0, n, n);
    fm::gesv(lhs_mass, dimension_components.back(), ipiv);
  }

  return combine_dimensions(degree, table, start, stop, dimension_components,
                            time_multiplier);
}

template<typename P>
inline void transform_and_combine_dimensions(
    std::vector<dimension<P>> const &dims,
    std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time,
    P const time_multiplier, fk::vector<P, mem_type::view> result)

{
  expect(v_functions.size() == static_cast<size_t>(dims.size()) or v_functions.size() == static_cast<size_t>(dims.size() + 1));
  expect(start <= stop);
  expect(stop < table.size());
  expect(degree >= 0);

  std::vector<fk::vector<P>> dimension_components;
  dimension_components.reserve(dims.size());

  for (size_t i = 0; i < dims.size(); ++i)
  {
    auto const &dim = dims[i];
    dimension_components.push_back(forward_transform<P>(
        dim, v_functions[i], dim.volume_jacobian_dV, transformer, time));
    int const n = dimension_components.back().size();
    std::vector<int> ipiv(n);
    fk::matrix<P> lhs_mass = dim.get_mass_matrix();
    expect(lhs_mass.nrows() == n);
    expect(lhs_mass.ncols() == n);
    fm::gesv(lhs_mass, dimension_components.back(), ipiv);
  }

  combine_dimensions(degree, table, start, stop, dimension_components,
                     time_multiplier, result);
}

template<typename P>
inline fk::vector<P> transform_and_combine_dimensions(
    std::vector<dimension<P>> const &dims,
    std::vector<vector_func<P>> const &v_functions,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const start, int const stop, int const degree, P const time = 0.0,
    P const time_multiplier = 1.0)
{
  int64_t const vector_size =
      (stop - start + 1) * fm::ipow(degree + 1, dims.size());
  expect(vector_size < INT_MAX);
  fk::vector<P> result(vector_size);
  transform_and_combine_dimensions(dims, v_functions, table, transformer, start,
                                   stop, degree, time, time_multiplier,
                                   fk::vector<P, mem_type::view>(result));
  return result;
}

template<typename P>
inline int dense_space_size(PDE<P> const &pde)
{
  return dense_space_size(pde.get_dimensions());
}

inline int dense_dim_size(int const degree, int const level)
{
  return (degree + 1) * fm::two_raised_to(level);
}

template<typename precision>
inline int dense_space_size(std::vector<dimension<precision>> const &dims)
{
  /* determine the length of the realspace solution */
  int64_t const dense_size = std::accumulate(
      dims.cbegin(), dims.cend(), int64_t{1},
      [](int64_t const size, dimension<precision> const &dim) {
        return size * dense_dim_size(dim.get_degree(), dim.get_level());
      });
  expect(dense_size <= std::numeric_limits<int>::max());
  return static_cast<int>(dense_size);
}

template<typename P>
using function_1d = std::function<void(std::vector<P> const &, std::vector<P> &)>;

/*!
 * \internal
 * \brief Stores mass-matrices per level
 *
 * A mass matrix has a block diagonal form where each block has size
 * (degree + 1) by (degree + 1).
 * At leach level, we have 2^l such blocks stored in a simple std::vector
 * so that we can handle the matrix operations using small-matrix algorithms.
 * The matrices are stored permanently so they can be reused.
 * \endinternal
 */
template<typename P>
struct mass_matrix
{
  //! returns the matrix entries at given level
  P *level(int l) { return mat[l].data(); }
  //! returns the matrix entries at given level (const)
  P const *level(int l) const { return mat[l].data(); }
  //! returns const matrix entries
  P const *clevel(int l) const { return mat[l].data(); }
  //! returns the matrix entries for given level
  void set(int l, std::vector<P> &&m) { mat[l] = std::move(m); }
  //! return true if the matrix has been set for this level
  bool has_level(int l) const { return (not mat[l].empty()); }
  //! matrices
  std::array<std::vector<P>, 31> mat;
  //! using list of mass matrices
  using list = std::array<std::unique_ptr<mass_matrix<P>>, max_num_dimensions>;
};

/*!
 * \internal
 * \brief Projects point-wise defined functions to the hierarchical basis
 *
 * The approach here is to reuse storage as much as possible, temporary workspace
 * is used with a few vectors and each is resized without shrinking the capacity.
 * Thus, allocations will be few.
 *
 * The methods use some side-effects to communicate, i.e., each method sets the
 * stage for the next method and the setup has to agree with assumptions.
 *
 * This class uses OpenMP and internal cache, so calls to any methods are not
 * thread-safe, except where OpenMP is already used internally.
 *
 * \endinternal
 */
template<typename P>
class hierarchy_manipulator
{
public:
  //! list of mass matrices, array with one unique_ptr per dimension
  using mass_list = typename mass_matrix<P>::list;

  //! empty hierarchy manipulator
  hierarchy_manipulator()
      : degree_(0), block_size_(0), dmin({{0}}), dmax({{0}})
  {}

  hierarchy_manipulator(int degree, int num_dimensions)
      : degree_(degree), block_size_(fm::ipow(degree + 1, num_dimensions)),
        dmin({{0}}), dmax({{1}}),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    setup_projection_matrices();
  }
  hierarchy_manipulator(int degree, int num_dimensions,
                        std::initializer_list<P> rmin,
                        std::initializer_list<P> rmax)
      : degree_(degree), block_size_(fm::ipow(degree + 1, num_dimensions)),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    expect(num_dimensions <= max_num_dimensions);
    std::copy_n(rmin.begin(), num_dimensions, dmin.begin());
    std::copy_n(rmax.begin(), num_dimensions, dmax.begin());
    setup_projection_matrices();
  }
  template<typename rangemin, typename rangemax>
  hierarchy_manipulator(int degree, int num_dimensions,
                        rangemin const &rmin, rangemax const &rmax)
      : degree_(degree), block_size_(fm::ipow(degree + 1, num_dimensions)),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    expect(num_dimensions <= max_num_dimensions);
    std::copy_n(rmin.begin(), num_dimensions, dmin.begin());
    std::copy_n(rmax.begin(), num_dimensions, dmax.begin());
    setup_projection_matrices();
  }
  hierarchy_manipulator(int degree, std::vector<dimension<P>> const &dims)
      : degree_(degree), block_size_(fm::ipow(degree + 1, dims.size())),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    for (auto i : indexof<int>(dims))
    {
      dmin[i] = dims[i].domain_min;
      dmax[i] = dims[i].domain_max;
    }
    setup_projection_matrices();
  }

  //! project separable function on the basis level
  template<data_mode action = data_mode::replace>
  void project_separable(P proj[],
                         std::vector<dimension<P>> const &dims,
                         std::vector<vector_func<P>> const &funcs,
                         std::vector<function_1d<P>> const &dv, mass_list &mass,
                         adapt::distributed_grid<P> const &grid,
                         P const time = 0.0, P const time_multiplier = 1.0,
                         int sstart = -1, int sstop = -1) const
  {
    // first we perform the one-dimensional transformations
    int const num_dims = static_cast<int>(dims.size());
    for (int d : indexof<int>(num_dims))
    {
      project1d([&](std::vector<P> const &x, std::vector<P> &fx)
          -> void {
        auto fkvec = funcs[d](x, time);
        std::copy(fkvec.begin(), fkvec.end(), fx.data());
      }, (dv.empty()) ? nullptr : dv[d], mass, d, dims[d].get_level());
    }

    // looking at row start and stop
    auto const &subgrid    = grid.get_subgrid(get_rank());
    int const *const cells = grid.get_table().get_active_table().data();

    if (sstart == -1)
    {
      sstart = subgrid.row_start;
      sstop  = subgrid.row_stop;
    }

    std::array<int, max_num_dimensions> midx;
    std::array<P const *, max_num_dimensions> data1d;

    int const pdof = degree_ + 1;
    for (int64_t s = sstart; s <= sstop; s++)
    {
      int const *const cc = cells + 2 * num_dims * s;
      asg2tsg_convert(num_dims, cc, midx.data());
      for (int d : indexof<int>(num_dims))
        data1d[d] = pf[d].data() + midx[d] * pdof;

      for (int64_t i : indexof(block_size_))
      {
        int64_t t = i;
        P val     = time_multiplier;
        for (int d = num_dims - 1; d >= 0; d--)
        {
          val *= data1d[d][t % pdof];
          t /= pdof;
        }
        if constexpr (action == data_mode::replace)
          proj[i] = val;
        else if constexpr (action == data_mode::increment)
          proj[i] += val;
      }

      proj += block_size_;
    }
  }

  //! computes the 1d projection of f onto the given level
  void project1d(function_1d<P> const &f, function_1d<P> const &dv,
                 mass_list &mass, int dim, int level) const
  {
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    fvals.resize(quad_points[dim].size()); // quad_points are resized and loaded above
    f(quad_points[dim], fvals);

    P const *m = nullptr;
    if (dv) // if using non-Cartesian coordinates
    {
      apply_dv_dvals(dim, dv);
      if (not mass[dim]->has_level(level))
        make_mass(dim, level, mass); // uses quad_dv computed above
      m = mass[dim]->clevel(level);
    }

    // project onto the basis
    project1d(dim, level, dmax[dim] - dmin[dim], m);
  }

  std::vector<P> const &get_projected1d(int dim) const { return pf[dim]; }

  int64_t block_size() const { return block_size_; }

protected:
  /*!
   * \brief Converts function values to the final hierarchical coefficients
   *
   * Assumes that fvals already contains the function values at the quadrature
   * points. The method will convert to local basis coefficients and then convert
   * to hierarchical representation stored in pf.
   */
  void project1d(int dim, int level, P const dsize, P const *mass) const;

  static constexpr P s2 = 1.41421356237309505; // std::sqrt(2.0)
  static constexpr P is2 = P{1} / s2;          // 1.0 / std::sqrt(2.0)
  static constexpr P s6 = 2.4494897427831781;  //std::sqrt(6.0)

  //! \brief Applies dv to the current fvals
  void apply_dv_dvals(int dim, function_1d<P> const &dv) const
  {
    if (quad_points[dim].size() != quad_dv[dim].size())
    {
      quad_dv[dim].resize(quad_points[dim].size());
      dv(quad_points[dim], quad_dv[dim]);
      for (auto i : indexof(quad_points[dim]))
        fvals[i] *= quad_dv[dim][i];
    }
  }
  //! \brief Constructs the mass matrix, if not set for the given level/dim (uses already set quad_dv)
  void make_mass(int dim, int level, mass_list &mass) const;

  /*!
   * \brief prepares the quad_points vector with the appropriate shifted quadrature points
   *
   * The quad_points can be used for a one shot call to the point-wise evaluation
   * function.
   */
  void prepare_quadrature(int dim, int num_cells) const;

  //! project 2 * num_final raw cells up the hierarchy into upper raw and final cells
  template<int degree>
  void projectup(int num_final, P const *raw, P *upper, P *fin) const;
  //! project the last two cells for level 0 and level 1
  template<int degree>
  void projectup2(P const *raw, P *fin) const;
  /*!
   * \brief Computes the local-coefficients to hierarchical representation
   *
   * The local coefficients must be already stored in stage0.
   * Both stage0 and stage1 will be used as scratch space here.
   */
  template<int degree>
  void projectlevels(int dim, int levels) const;

  //! call from the constructor, makes it easy to have variety of constructor options
  void setup_projection_matrices();

private:
  int degree_;
  int64_t block_size_;

  std::array<P, max_num_dimensions> dmin, dmax;

  static int constexpr points  = 0;
  static int constexpr weights = 1;
  vector2d<P> quad; // single cell quadrature
  vector2d<P> leg_vals; // values of Legendre polynomials at the quad points
  vector2d<P> leg_unscal; // Legendre polynomials not-scaled by the quadrature w.

  std::vector<P> pmats; // projection matrices
  P *pmatup  = nullptr; // this to upper level (alias to pmats)
  P *pmatlev = nullptr; // this to same level (alias to pmats)

  // given the values of f(x) at the quadrature points inside of a cell
  // the projection of f onto the Legendre basis is leg_vals * f
  // i.e., small matrix times a small vector

  mutable std::array<std::vector<P>, max_num_dimensions> pf;
  mutable std::array<std::vector<P>, max_num_dimensions> quad_points;
  mutable std::array<std::vector<P>, max_num_dimensions> quad_dv;
  mutable std::vector<P> fvals;
  mutable std::vector<P> stage0, stage1;
};


} // namespace asgard
