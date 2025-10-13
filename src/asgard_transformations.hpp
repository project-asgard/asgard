#pragma once
#include "asgard_kronmult.hpp"
#include "asgard_pde.hpp"
#include "asgard_pde_functions.hpp"
#include "asgard_wavelet_basis.hpp"

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

//! function format for 1d function, allows evaluations for large batches of funcitons
template<typename P>
using function_1d = std::function<void(std::vector<P> const &, std::vector<P> &)>;

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

  //! fill a vector with the interior quadrature shifted to the domain
  void interior_quad(P xleft, P xright, int level, std::vector<P> &pnts);

  //! raw comes from coefficient-mats and holds the values of rhs at quad points
  std::vector<P> project(bool is_interior, int level, P dsqrt, P alpha,
                         std::vector<P> const &raw) const;
  //! raw2 is multiplied by raw1 and then those are projected as above
  std::vector<P> project(bool is_interior, int level, P dsqrt,
                         std::vector<P> const &raw1, std::vector<P> &raw2) const;
  //! project the constant onto the legendre basis
  std::vector<P> project(int level, P dsqrt, P alpha) const;
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
  /*!
   * \brief Indicated whether to use standard wavelet transform or permutation
   *
   * The algorithms for both operations are identical, the difference
   * is in the matrices being used.
   * The enum are used in conjunction with if-constexpr to select
   * the proper matrices to apply.
   */
  enum class operation {
    //! transform cell-by-cell Legendre basis to hierarchical wavelets
    transform,
    //! perform custom unitary transformation using a custom matrix
    custom_unitary,
    //! perform custom non-unitary transformation using a custom matrix
    custom_non_unitary,
  };

  //! empty hierarchy manipulator
  hierarchy_manipulator()
      : degree_(0), block_size_(0)
  {
    std::fill(dmin.begin(), dmin.end(), 0);
    std::fill(dmax.begin(), dmax.end(), 0);
  }
  //! set the degree and number of dimensions
  hierarchy_manipulator(int degree, int num_dimensions)
      : degree_(degree), block_size_(fm::ipow(degree + 1, num_dimensions)),
        quad(make_quadrature<P>(2 * degree_ + 1, -1, 1))
  {
    std::fill(dmin.begin(), dmin.end(), 0);
    std::fill(dmax.begin(), dmax.end(), 1);
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
  void project_separable(separable_func<P> const &sep,
                         sparse_grid const &grid, mass_diag<P> const &mass,
                         P time, P alpha, P f[]) const;

  //! computes the 1d projection of f onto the given level, result is in get_projected1d(dim)
  void project1d_f(function_1d<P> const &f, block_diag_matrix<P> const &mass, int dim, int level,
                   std::vector<P> &proj_f) const
  {
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    fvals.resize(quad_points[dim].size()); // quad_points are resized and loaded above
    f(quad_points[dim], fvals);

    project1d(dim, level, fvals, mass, pwork);
    transform(level, pwork, proj_f);
  }
  //! computes the 1d projection of f onto the given level, result is in get_projected1d(dim)
  std::vector<P> get_project1d_f(function_1d<P> const &f, block_diag_matrix<P> const &mass,
                                 int dim, int level) const
  {
    std::vector<P> result;
    project1d_f(f, mass, dim, level, result);
    return result;
  }
  //! computes the 1d projection of constant onto the given level, result is in get_projected1d(dim)
  void project1d_c(P const c, block_diag_matrix<P> const &mass, int dim, int level,
                   std::vector<P> &proj_f) const
  {
    int const num_cells = fm::ipow2(level);
    if (mass) {
      fvals.resize(num_cells * quad.stride());
      std::fill(fvals.begin(), fvals.end(), c);

      project1d(dim, level, fvals, mass, pwork);
      transform(level, pwork, proj_f);
    } else {
      // the projection is trivial, exploiting orthogonality of the basis
      proj_f.resize(num_cells * (degree_ + 1));
      proj_f.front() = c * std::sqrt(dmax[dim] - dmin[dim]);
      std::fill(proj_f.begin() + 1, proj_f.end(), 0);
    }
  }
  //! computes the 1d projection of constant onto the given level, result is in get_projected1d(dim)
  std::vector<P> get_project1d_c(P const c, block_diag_matrix<P> const &mass, int dim, int level) const
  {
    std::vector<P> result;
    project1d_c(c, mass, dim, level, result);
    return result;
  }

  //! (testing purposes, skips hierarchy) computes the 1d projection of f onto the cells of a given level
  std::vector<P> cell_project(int dim, function_1d<P> const &f, int level) const
  {
    int const num_cells = fm::ipow2(level);
    prepare_quadrature(dim, num_cells);
    fvals.resize(quad_points[dim].size()); // quad_points are resized and loaded above
    f(quad_points[dim], fvals);

    // project onto the basis
    std::vector<P> result;
    project1d(dim, level, fvals, block_diag_matrix<P>{}, result);
    return result;
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

  //! converts matrix from diagonal to transformed on left/right with the given operations
  block_sparse_matrix<P> diag2block(
      operation left, P const tl[], operation right, P const tr[],
      block_diag_matrix<P> const &diag,
      int const level, connection_patterns const &conns) const
  {
    block_sparse_matrix<P> col = make_block_sparse_matrix(conns, connect_1d::hierarchy::col_volume);
    block_sparse_matrix<P> res = make_block_sparse_matrix(conns, connect_1d::hierarchy::volume);

    do_col_project_vol(right, tr, diag, level, conns, col);
    do_row_project_any(left, tl, col, level, conns, res);

    return res;
  }

  //! converts matrix from diagonal to transformed on left/right with the given operations
  block_sparse_matrix<P> tri2block(
      operation left, P const tl[], operation right, P const tr[],
      block_tri_matrix<P> const &tri,
      int const level, connection_patterns const &conns) const
  {
    block_sparse_matrix<P> col = make_block_sparse_matrix(conns, connect_1d::hierarchy::col_full);
    block_sparse_matrix<P> res = make_block_sparse_matrix(conns, connect_1d::hierarchy::full);

    do_col_project_full(right, tr, tri, level, conns, col);
    do_row_project_any(left, tl, col, level, conns, res);

    return res;
  }

  //! transform cell-by-cell Legendre coefficients into hierarchical wavelet coefficients
  void transform(int level, P src[], P dest[]) const
  {
    constexpr operation op = operation::transform;
    switch (degree_) {
      case 0:
        apply_transform<0, op>(level, src, dest);
        break;
      case 1:
        apply_transform<1, op>(level, src, dest);
        break;
      default:
        apply_transform<-1, op>(level, src, dest);
        break;
    };
  }
  //! transform with vector overload
  void transform(int level, std::vector<P> &src, std::vector<P> &dest) const
  {
    expect(static_cast<int64_t>(src.size()) == fm::ipow2(level) * (degree_ + 1));
    dest.resize(src.size());
    transform(level, src.data(), dest.data());
  }
  //! transforms the vector to a hierarchical representation
  void transform(int const level, std::vector<P> &x) const
  {
    if (level == 0) // nothing to project at level 0
      return;
    int64_t const size = fm::ipow2(level) * (degree_ + 1);
    expect(size == static_cast<int64_t>(x.size()));

    pwork.resize(size);
    std::copy_n(x.begin(), size, pwork.begin());
    transform(level, pwork.data(), x.data());
  }

  //! apply a custom transform to the vectors (works for both unitary and non-unitary)
  void transform(P const *trans, int level, P src[], P dest[]) const
  {
    // the unitary/non-unitary property of the map relates only to the inverse,
    // i.e., when constructing the column transformation
    expect(trans != nullptr);
    constexpr operation op = operation::custom_unitary;
    switch (degree_) {
      case 0:
        apply_transform<0, op>(trans, level, src, dest);
        break;
      case 1:
        apply_transform<1, op>(trans, level, src, dest);
        break;
      default:
        apply_transform<-1, op>(trans, level, src, dest);
        break;
    };
  }
  //! transform with vector overload
  void transform(P const *trans, int level, std::vector<P> &src, std::vector<P> &dest) const
  {
    expect(static_cast<int64_t>(src.size()) == fm::ipow2(level) * (degree_ + 1));
    dest.resize(src.size());
    transform(trans, level, src.data(), dest.data());
  }

protected:
  /*!
   * \brief Perform the transformation on the given data
   *
   * \tparam tdegree is the the degree, allows hardcoding simple matrices
   *
   * \param level is the level for the transformation
   * \param src is the source with size 2^level, this operation will destroy the source
   * \param dest is the destination with same size as src
   */
  template<int tdegree, operation op>
  void apply_transform(int level, P src[], P dest[]) const {
    apply_transform<tdegree, op>(nullptr, level, src, dest);
  }

  template<int tdegree, operation op>
  void apply_transform(P const *trans, int level, P src[], P dest[]) const;

  //! Given values of a function, project on the cell-by-cell basis
  void project1d(int dim, int level, std::vector<P> const &vals,
                 block_diag_matrix<P> const &mass, std::vector<P> &cells) const;
  //! reusable constants std::sqrt(2.0)
  static constexpr P s2 = 1.41421356237309505;
  //! reusable constants 1.0 / std::sqrt(2.0)
  static constexpr P is2 = P{1} / s2;
  //! reusable constants std::sqrt(6.0)
  static constexpr P s6 = 2.4494897427831781;

  /*!
   * \brief prepares the quad_points vector with the appropriate shifted quadrature points
   *
   * The quad_points can be used for a one shot call to the point-wise evaluation
   * function.
   */
  void prepare_quadrature(int dim, int num_cells) const;

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
  template<int tdegree, operation op>
  void col_project_full(P const *trans,
                        block_tri_matrix<P> const &tri,
                        int const level,
                        connection_patterns const &conn,
                        block_sparse_matrix<P> &sp) const;

  //! apply column transform on tri-diagonal matrix -> sparse in col-full pattern
  template<int tdegree, operation op>
  void col_project_vol(P const *trans,
                       block_diag_matrix<P> const &diag,
                       int const level,
                       connection_patterns const &conn,
                       block_sparse_matrix<P> &sp) const;

  //! apply row transform on sparse col-full pattern
  template<int tdegree, operation op>
  void row_project_any(P const *trans,
                       block_sparse_matrix<P> &col,
                       int const level,
                       connection_patterns const &conn,
                       block_sparse_matrix<P> &sp) const;

  //! maps the degree for the specified operation
  template<operation op>
  void do_col_project_full_(P const trans[],
                            block_tri_matrix<P> const &tri,
                            int const level,
                            connection_patterns const &conn,
                            block_sparse_matrix<P> &sp) const
  {
    switch (degree_) {
    case 0:
      col_project_full<0, op>(trans, tri, level, conn, sp);
      break;
    case 1:
      col_project_full<1, op>(trans, tri, level, conn, sp);
      break;
    default:
      col_project_full<-1, op>(trans, tri, level, conn, sp);
      break;
    };
  }
  //! maps the operation to the correct template and degree
  void do_col_project_full(operation op, P const trans[],
                           block_tri_matrix<P> const &tri,
                           int const level,
                           connection_patterns const &conn,
                           block_sparse_matrix<P> &sp) const
  {
    switch (op) {
    case operation::custom_unitary:
      do_col_project_full_<operation::custom_unitary>(trans, tri, level, conn, sp);
      break;
    case operation::custom_non_unitary:
      do_col_project_full_<operation::custom_non_unitary>(trans, tri, level, conn, sp);
      break;
    default: // case operation::transform:
      do_col_project_full_<operation::transform>(trans, tri, level, conn, sp);
      break;
    };
  }

  //! maps the degree for the specified operation
  template<operation op>
  void do_col_project_vol_(P const trans[],
                           block_diag_matrix<P> const &diag,
                           int const level,
                           connection_patterns const &conn,
                           block_sparse_matrix<P> &sp) const
  {
    switch (degree_) {
    case 0:
      col_project_vol<0, op>(trans, diag, level, conn, sp);
      break;
    case 1:
      col_project_vol<1, op>(trans, diag, level, conn, sp);
      break;
    default:
      col_project_vol<-1, op>(trans, diag, level, conn, sp);
      break;
    };
  }
  //! maps the operation to the correct template and degree
  void do_col_project_vol(operation op, P const trans[],
                          block_diag_matrix<P> const &diag,
                          int const level,
                          connection_patterns const &conn,
                          block_sparse_matrix<P> &sp) const
  {
    switch (op) {
    case operation::custom_unitary:
      do_col_project_vol_<operation::custom_unitary>(trans, diag, level, conn, sp);
      break;
    case operation::custom_non_unitary:
      do_col_project_vol_<operation::custom_non_unitary>(trans, diag, level, conn, sp);
      break;
    default: // case operation::transform:
      do_col_project_vol_<operation::transform>(trans, diag, level, conn, sp);
      break;
    };
  }

  //! maps the degree for the specified operation
  template<operation op>
  void do_row_project_any_(P const trans[],
                           block_sparse_matrix<P> &col,
                           int const level,
                           connection_patterns const &conn,
                           block_sparse_matrix<P> &sp) const
  {
    switch (degree_) {
    case 0:
      row_project_any<0, op>(trans, col, level, conn, sp);
      break;
    case 1:
      row_project_any<1, op>(trans, col, level, conn, sp);
      break;
    default:
      row_project_any<-1, op>(trans, col, level, conn, sp);
      break;
    };
  }
  //! maps the operation to the correct template and degree
  void do_row_project_any(operation op, P const trans[],
                          block_sparse_matrix<P> &col,
                          int const level,
                          connection_patterns const &conn,
                          block_sparse_matrix<P> &sp) const
  {
    if (op == operation::transform) {
      do_row_project_any_<operation::transform>(trans, col, level, conn, sp);
    } else {
      // the unitary and the non-unitary transforms are equivalent here
      do_row_project_any_<operation::custom_unitary>(trans, col, level, conn, sp);
    }
  }

  //! call from the constructor, makes it easy to have variety of constructor options
  void setup_projection_matrices();

private:
  int degree_;
  int64_t block_size_;

  std::array<P, max_num_dimensions> dmin, dmax;

  static int constexpr points  = 0; // tags for the entries in the quadrature structure
  static int constexpr weights = 1;
  vector2d<P> quad; // single cell quadrature
  vector2d<P> leg_vals; // values of Legendre polynomials at the quad points
  vector2d<P> leg_unscal; // Legendre polynomials not-scaled by the quadrature w.

  std::vector<P> tmats; // transformation matrices
  P *tmatup  = nullptr; // this to upper level (alias to tmats)
  P *tmatlev = nullptr; // this to same level (alias to tmats)

  // given the values of f(x) at the quadrature points inside of a cell
  // the projection of f onto the Legendre basis is leg_vals * f
  // i.e., small matrix times a small vector

  // projected function values for each dimension
  mutable std::array<std::vector<P>, max_num_dimensions> pf;
  // quadrature points workspace for each direction
  mutable std::array<std::vector<P>, max_num_dimensions> quad_points;
  // workspace for function values at quadrature nodes
  mutable std::vector<P> fvals;
  // workspaces for projection and transformation
  mutable std::vector<P> pwork, twork;

  mutable std::vector<std::vector<P>> colblocks;
  // TODO: make reusable cache matrixes
  //mutable std::array<block_sparse_matrix<P>, 4> rowstage;
};

} // namespace asgard
