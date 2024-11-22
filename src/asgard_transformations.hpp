#pragma once
#include "asgard_batch.hpp"
#include "asgard_kron_operators.hpp"

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
  int const n                  = fm::ipow2(num_levels);
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
inline int dense_space_size(PDE<P> const &pde)
{
  return dense_space_size(pde.get_dimensions());
}

inline int dense_dim_size(int const degree, int const level)
{
  return (degree + 1) * fm::ipow2(level);
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
 * This class uses OpenMP and internal cache, so calls to all methods are not
 * thread-safe.
 *
 * (eventually, this will need cleanup of the api calls but right now the focus
 *  is on performance and capability)
 *
 * \endinternal
 */
template<typename P>
class hierarchy_manipulator
{
public:
  //! list of mass matrices, array with one unique_ptr per dimension
  using mass_list = std::array<level_mass_matrces<P>, max_num_dimensions>;

  //! empty hierarchy manipulator
  hierarchy_manipulator()
      : degree_(0), block_size_(0), dmin({{0}}), dmax({{0}})
  {}
  //! set the degree and number of dimensions
  hierarchy_manipulator(int degree, int num_dimensions)
      : degree_(degree), block_size_(fm::ipow(degree + 1, num_dimensions)),
        dmin({{0}}), dmax({{1}}),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    setup_projection_matrices();
  }
  //! initialize with the given domain
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
  //! flexibile initialize, randes are defined in array-like objects
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
  //! initialize form the given set of dimensions
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
                         std::array<function_1d<P>, max_num_dimensions> const &dv,
                         mass_list &mass,
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
      }, (dv.empty()) ? nullptr : dv[d], mass[d], d, dims[d].get_level());
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
                 level_mass_matrces<P> &mass, int dim, int level) const
  {
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    fvals.resize(quad_points[dim].size()); // quad_points are resized and loaded above
    f(quad_points[dim], fvals);

    if (dv) // if using non-Cartesian coordinates
    {
      apply_dv_dvals(dim, dv);
      mass.set_non_identity();
      if (not mass.has_level(level))
        mass[level] = make_mass(dim, level); // uses quad_dv computed above
    }

    // project onto the basis
    project1d(dim, level, dmax[dim] - dmin[dim], mass);
  }

  //! create the mass matrix for the given dim and level
  void make_mass(int dim, int level, function_1d<P> const &dv,
                 level_mass_matrces<P> &mass) const
  {
    if (not dv or mass.has_level(level))
      return;
    mass.set_non_identity();
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    quad_dv[dim].resize(quad_points[dim].size());
    dv(quad_points[dim], quad_dv[dim]);
    mass[level] = make_mass(dim, level); // uses quad_dv computed above
  }

  //! return the 1d projection in the given direction
  std::vector<P> const &get_projected1d(int dim) const { return pf[dim]; }

  //! transforms the vectors to hierarchical representation
  void project1d(int const level, fk::vector<P> &x) const
  {
    int64_t const size = fm::ipow2(level) * (degree_ + 1);
    expect(size == x.size());
    stage0.resize(size);
    pf[0].resize(size);
    std::copy_n(x.begin(), size, stage0.begin());
    switch (degree_)
    { // hardcoded degrees first, the default uses the projection matrices
    case 0:
      projectlevels<0>(0, level);
      break;
    case 1:
      projectlevels<1>(0, level);
      break;
    default:
      projectlevels<-1>(0, level);
    };
    std::copy_n(pf[0].begin(), size, x.begin());
  }

  //! size of a multi-dimensional block, i.e., (degree + 1)^d
  int64_t block_size() const { return block_size_; }
  //! returns the degree
  int degree() const { return degree_; }

  //! converts matrix from tri-diagonal to hierarchical sparse format
  block_sparse_matrix<P> tri2hierarchical(
      block_tri_matrix<P> const &tri, int const level, connection_patterns const &conns) const;
  //! converts matrix from diagonal to hierarchical sparse format
  block_sparse_matrix<P> diag2hierarchical(
      block_diag_matrix<P> const &diag, int const level, connection_patterns const &conns) const;

protected:
  /*!
   * \brief Converts function values to the final hierarchical coefficients
   *
   * Assumes that fvals already contains the function values at the quadrature
   * points. The method will convert to local basis coefficients and then convert
   * to hierarchical representation stored in pf.
   */
  void project1d(int dim, int level, P const dsize, level_mass_matrces<P> const &mass) const;

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
  mass_matrix<P> make_mass(int dim, int level) const;

  /*!
   * \brief prepares the quad_points vector with the appropriate shifted quadrature points
   *
   * The quad_points can be used for a one shot call to the point-wise evaluation
   * function.
   */
  void prepare_quadrature(int dim, int num_cells) const;

  //! project 2 * num_final raw cells up the hierarchy into upper raw and final cells
  template<int tdegree>
  void projectup(int num_final, P const *raw, P *upper, P *fin) const;
  //! project the last two cells for level 0 and level 1
  template<int tdegree>
  void projectup2(P const *raw, P *fin) const;
  /*!
   * \brief Computes the local-coefficients to hierarchical representation
   *
   * The local coefficients must be already stored in stage0.
   * Both stage0 and stage1 will be used as scratch space here.
   */
  template<int tdegree>
  void projectlevels(int dim, int levels) const;

  //! creates a new sparse matrix with the given format
  block_sparse_matrix<P> make_block_sparse_matrix(connection_patterns const &conns,
                                                  connect_1d::hierarchy const h) const
  {
    return block_sparse_matrix<P>((degree_ + 1) * (degree_ + 1), conns(h).num_connections(), h);
  }

  //! apply column transform on tri-diagonal matrix -> sparse in col-full pattern
  template<int tdegree>
  void col_project_full(block_tri_matrix<P> const &tri,
                        int const level,
                        connection_patterns const &conn,
                        block_sparse_matrix<P> &sp) const;

  //! apply column transform on tri-diagonal matrix -> sparse in col-full pattern
  template<int tdegree>
  void col_project_full(block_diag_matrix<P> const &diag,
                        int const level,
                        connection_patterns const &conn,
                        block_sparse_matrix<P> &sp) const;

  //! apply row transform on sparse col-full pattern
  template<int tdegree>
  void row_project_full(block_sparse_matrix<P> &col,
                        int const level,
                        connection_patterns const &conn,
                        block_sparse_matrix<P> &sp) const;

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

  mutable std::array<block_matrix<P>, 2> matstage;

  mutable std::vector<std::vector<P>> colblocks;
  mutable std::array<block_sparse_matrix<P>, 4> rowstage;
};

} // namespace asgard
