#pragma once
#include "asgard_adapt.hpp"
#include "asgard_kron_operators.hpp"

namespace asgard
{
/*!
 * \internal
 * \brief Shorthand for array of diagonal mass matrices
 *
 * \endinternal
 */
template<typename P>
using mass_diag = std::array<block_diag_matrix<P>, max_num_dimensions>;

// combines the values from the vectors into the combined tensor list
// the size of combined should be equal to the number of elements
// times the tesor block size (degree + 1)^d
template<typename P>
void combine_dimensions(int const degree, elements::table const &table,
                        int const start_element, int const stop_element,
                        std::vector<std::vector<P>> const &vectors,
                        P combined[]);

/*!
 * \internal
 * \brief Legendre basis, quadrature, polynomial and derivative values
 *
 * The entries are used to construct the coefficient matrices.
 * \endinternal
 */
template<typename P>
struct legendre_basis {
  //! create empty basis, nothing is initialized and this will have to be reinitialized
  legendre_basis() = default;
  //! construct a basis for this degree
  legendre_basis(int degree);
  //! polynomial degree of freedom, i.e., degree + 1
  int pdof = 0;
  //! number of quadrature points
  int num_quad = 0;
  //! all data in a single spot
  std::vector<P> data_;

  //! quadrature points
  P *qp = nullptr;
  //! quadrature weights
  P *qw = nullptr;

  //! flux, from self across left edge
  P *to_left = nullptr; // scale 1 / dx
  //! flux, from the left cell
  P *from_left = nullptr;
  //! flux, from the right cell
  P *from_right = nullptr;
  //! flux, from self across right edge
  P *to_right = nullptr;

  //! legendre polynomials evaluated at the quadrature points
  P *leg = nullptr; // scale 1 / sqrt(dx)
  //! legendre polynomials evaluated at the quadrature points and scaled by the quadrature weights
  P *legw = nullptr; // scale 1 / sqrt(dx)
  //! legendre derivatives evaluated at the quadrature points
  P *der = nullptr; // scale 1 / (2 * dx * sqrt(dx))

  //! values of the legendre polynomials at the left end-point
  P *leg_left = nullptr;
  //! values of the legendre polynomials at the right end-point
  P *leg_right = nullptr;

  void project(bool is_interior, int level, std::vector<P> const &raw,
               std::vector<P> &lgn) const;
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
  //! initialize form the given set of dimensions
  hierarchy_manipulator(int degree, pde_domain<P> const &domain)
      : degree_(degree), block_size_(fm::ipow(degree + 1, domain.num_dims())),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    for (int i : iindexof(domain.num_dims()))
    {
      dmin[i] = domain.xleft(i);
      dmax[i] = domain.xright(i);
    }
    setup_projection_matrices();
  }

  //! project separable function on the basis level
  template<data_mode action = data_mode::replace>
  void project_separable(separable_func<P> const &sep, pde_domain<P> const &domain,
                         sparse_grid const &grid, mass_diag<P> const &mass,
                         P time, P alpha, P f[]) const
  {
    int const num_dims = domain.num_dims();
    for (int d : iindexof(num_dims))
    {
      if (sep.is_const(d)) {
        project1d_c(sep.cdomain(d), mass[d], d, grid.current_level(d));
      } else {
        project1d_f([&](std::vector<P> const &x, std::vector<P> &fx)
            -> void {
          sep.fdomain(d, x, time, fx);
        }, mass[d], d, grid.current_level(d));
      }
    }

    P const tmult = (sep.ftime()) ? sep.ftime()(time) : P{1};

    int const pdof = degree_ + 1;
    std::array<P const *, max_num_dimensions> data1d;

    P *proj = f;

    for (auto c : indexof(grid.num_indexes()))
    {
      int const *idx = grid[c];
      for (int d : iindexof(num_dims))
        data1d[d] = pf[d].data() + idx[d] * pdof;

      for (int64_t i : indexof(block_size_))
      {
        int64_t t = i;
        P val     = tmult;
        for (int d = num_dims - 1; d >= 0; d--) {
          val *= data1d[d][t % pdof];
          t /= pdof;
        }
        if constexpr (action == data_mode::replace)
          proj[i] = val;
        else if constexpr (action == data_mode::scal_rep)
          proj[i] = alpha * val;
        else if constexpr (action == data_mode::increment)
          proj[i] += val;
        else if constexpr (action == data_mode::scal_inc)
          proj[i] += alpha * val;
      }

      proj += block_size_;
    }
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
    static_assert(action == data_mode::replace or action == data_mode::increment);
    // first we perform the one-dimensional transformations
    int const num_dims = static_cast<int>(dims.size());
    for (int d : indexof<int>(num_dims))
    {
      project1d_f([&](std::vector<P> const &x, std::vector<P> &fx)
          -> void {
        auto fkvec = funcs[d](x, time);
        std::copy(fkvec.begin(), fkvec.end(), fx.data());
      }, dv[d], mass[d], d, dims[d].get_level());
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
      asg2tsg_convert(num_dims, cells + 2 * num_dims * s, midx.data());
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

  //! computes the 1d projection of f onto the given level, result is in get_projected1d(dim)
  void project1d_f(function_1d<P> const &f, function_1d<P> const &dv,
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
  //! computes the 1d projection of f onto the given level, result is in get_projected1d(dim)
  void project1d_f(function_1d<P> const &f, block_diag_matrix<P> const &mass, int dim, int level) const
  {
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    fvals.resize(quad_points[dim].size()); // quad_points are resized and loaded above
    f(quad_points[dim], fvals);

    project1d(dim, level, dmax[dim] - dmin[dim], mass);
  }
  //! computes the 1d projection of f onto the given level, result is in get_projected1d(dim)
  std::vector<P> get_project1d_f(function_1d<P> const &f, block_diag_matrix<P> const &mass, int dim, int level) const
  {
    project1d_f(f, mass, dim, level);
    return get_projected1d(dim);
  }
  //! computes the 1d projection of constant onto the given level, result is in get_projected1d(dim)
  void project1d_c(P const c, block_diag_matrix<P> const &mass, int dim, int level) const
  {
    int const num_cells = fm::ipow2(level);
    if (mass) {
      fvals.resize(num_cells * quad.stride());
      std::fill(fvals.begin(), fvals.end(), c); // TODO: skip the projection below
      project1d(dim, level, dmax[dim] - dmin[dim], mass);
    } else {
      // the projection is trivial, exploiting orthogonality of the basis
      pf[dim].resize(num_cells * (degree_ + 1));
      std::fill(pf[dim].begin(), pf[dim].end(), 0);
      pf[dim].front() = c;
    }
  }
  //! computes the 1d projection of constant onto the given level, result is in get_projected1d(dim)
  std::vector<P> get_project1d_c(P const c, block_diag_matrix<P> const &mass, int dim, int level) const
  {
    project1d_c(c, mass, dim, level);
    return get_projected1d(dim);
  }

  //! (testing purposes, skips hierarchy) computes the 1d projection of f onto the cells of a given level
  std::vector<P> cell_project(function_1d<P> const &f, function_1d<P> const &dv, int level) const
  {
    int constexpr dim = 0;
    level_mass_matrces<P> mass;

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
    bool constexpr skip_hier = true;
    project1d<skip_hier>(dim, level, dmax[dim] - dmin[dim], mass);

    return stage0;
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

  //! transforms the vector to a hierarchical representation
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
  //! transforms the vector to a hierarchical representation
  void project1d(int const level, std::vector<P> &x) const
  {
    if (level == 0) // nothing to project at level 0
      return;
    int64_t const size = fm::ipow2(level) * (degree_ + 1);
    expect(size == static_cast<int64_t>(x.size()));
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
  //! transform the batch of vectors to nodal representation
  void reconstruct1d(int const nbatch, int const level, span2d<P> hdata) const;

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
  template<bool skip_hierarchy = false>
  void project1d(int dim, int level, P const dsize, level_mass_matrces<P> const &mass) const;
  //! project onto the basis
  void project1d(int dim, int level, P const dsize, block_diag_matrix<P> const &mass) const;

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

  //! tempalted version for reduction of runtime if-statements
  template<int tdegree>
  void reconstruct1d(int const nbatch, int level, span2d<P> data) const;

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
