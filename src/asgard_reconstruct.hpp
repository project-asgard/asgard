#pragma once

#include "asgard_interpolation.hpp"

/*!
 * \file asgard_reconstruct.hpp
 * \brief Provides the class for reconstructing solution at arbitrary points in the domain.
 * \author The ASGarD Team
 * \ingroup asgard_postprocess
 */

namespace asgard
{

/*!
 * \defgroup asgard_postprocess ASGarD Post Processing Tools
 *
 * The internal data-structures of ASGarD store the solution state in hierarchical
 * format, i.e., vector of coefficients associated with sparse grid points.
 * Interpretation of the data in raw format is challenging at best but converting
 * the data into a more standard dense-grid format is also not feasible in
 * high dimensions.
 *
 * The asgard::reconstruct_solution can find the value of the solution at
 * arbitrary points of the domain but computing the sum of coefficients times
 * the values of the basis functions. The class also comes with bindings for
 * Python and MATLAB. See the included examples.
 *
 * \internal
 * The C bindings allow Python to hold onto a reconstruct_solution object via a
 * void pointer. On every call, reinterpret_cast is used on the pointer to match
 * the correct type and a call to the corresponding member method is made.
 * \endinternal
 */

/*!
 * \ingroup asgard_postprocess
 * \brief Reconstruct the solution at arbitrary points in the domain from hierarchical data
 *
 * The class is intended to be used from within Python for plotting
 * and post-processing of the data. The API uses raw-arrays to facilitate
 * interfacing with the Python ctypes module.
 *
 * The reconstruction algorithms are taken from Tasmanian.
 */
class reconstruct_solution
{
public:
  /*!
   * \brief Constructor, prepares the data-structures for reconstruction
   *
   * \tparam precision is auto-directed as float or double, reconstruction
   *                   uses double precision but the coefficients can be
   *                   provided in single precision
   *
   * \param dims is the problem number of dimenisons
   * \param num_cells is the number of sparse grid cells (multi-indexes)
   * \param asg_cells is an array of size 2 * dims * num_cells containing
   *                  the multi-indexes in ASGarD format (level, index)
   * \param degree is the polynomial order
   * \param solution is the current set of coefficients representing
   *                 the PDE solution, size is (degree + 1)^dims * num_cells
   */
  template<typename precision>
  reconstruct_solution(int dims, int64_t num_cells, int const asg_cells[],
                       int degree, precision const solution[]);

  /*!
   * \brief Sets the min/max bound of the domain
   *
   * Both arrays should have size equal to the number of dimensions,
   * without this call the domain is set to the canonical (0, 1)
   */
  void set_domain_bounds(double const amin[], double const amax[]);

  /*!
   * \brief Reconstruct the values of the solution at the given points
   *
   * \param x has size num_x * number-of-dimensions and each strip of dims
   *          entries holds one point in the domain
   * \param num_x is the number of points for the reconstruction
   * \param y has size num_x and will hold the values of the solution
   *          at the given set of points
   */
  void reconstruct(double const x[], int num_x, double y[]) const;

  //! Returns the number of dimensions
  int num_dimensions() const { return cells_.num_dimensions(); }
  //! Returns the number of SG cells
  int64_t num_cells() const { return cells_.num_indexes(); }

  /*!
   * \brief Returns the centers of the sparse grid cells
   */
  void cell_centers(double x[]) const;

protected:
#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL
  //! template dispatch based on the degree
  template<int degree>
  void reconstruct(double const x[], int num_x, double y[]) const;

  /*!
   * \brief Prepare the inernal data-structures for fast reconstruction
   *
   * The way things are commonly done in ASGarD is to build the basis on a full
   * grid (at least in 1D), then move to wavelet space using a matrix product.
   * The process is expensive and becomes less viable after level 10.
   *
   * The alternative is to find analytic expressions for the wavelets and avoid
   * going to "real" or "non-wavelet" space
   * and because the basis is local, many of the functions can be
   * skipped in the reconstruction. The sparse grids cells natually form
   * a directed-acyclic-graph (DAG) and we can use the fact that the support
   * of the children is included in the support of the parent, so if the parent
   * is not supported over a given point in the domain, all children are
   * also non-supported and can be skipepd in the evaluation.
   *
   * On the other hand, since the DAG allows for multiple parents, a standard
   * tree traversal algorithm will double-count some of the the supported
   * functions.  This method replaces the DAG with a tree structure by
   * removing redundant connections.
   * The main goals is to simplify finding the basis functions supported over
   * a given point.
   * There may be multiple trees due to orphan nodes in the refinement process,
   * but every tree has a very simple form.
   *
   * The tree (or trees) are represented by a sparse matrix structure where
   * each non-zero represents a parent-child relationship in the tree
   * structure.
   */
  void build_tree();

  //! Helper method that finds the children of all multi-indexes
  vector2d<int> compute_dag_down() const;
  //! Helper method that computes the level for each multi-index
  std::vector<int> compute_levels() const;
  /*!
   * \brief Find the values of the basis functions
   *
   * For the cell p, it computes the basis functions at x[] and multiplies
   * those times the coefficients c.
   * At this stage of the algorithm, the point x has to be defined over the
   * canonical domain.
   *
   * \param p is the multi-index for the sparse grid cell
   * \param x is the point of interest
   * \param c is the coefficients of the basis functions in this cell
   *
   * \returns the value of the sum of basis times coefficients,
   * the optional is empty if the basis is not supported.
   */
  template<int degree>
  std::optional<double>
  basis_value(int const p[], double const x[], double const c[],
              vector2d<double> &scratch) const;

  /*!
   * \brief Traverses the graph and returns the sum of coefficients times basis functions
   */
  template<int degree>
  double walk_trees(const double x[]) const;
#endif

private:
  int pterms_;
  int64_t block_size_;

  indexset cells_;
  std::vector<double> coeff_;

  std::array<double, max_num_dimensions> inv_slope, shift;
  double domain_scale;

  // tree for evaluation
  std::vector<int> roots;
  std::vector<int> pntr;
  std::vector<int> indx;

  // using singe vector so data is compact and cached more easily
  // alias the location of the left-right basis
  std::vector<double> wavelets;
  // aliases to the structure above
  double *wleft  = nullptr;
  double *wright = nullptr;
};

} // namespace asgard

#ifndef __ASGARD_DOXYGEN_SKIP_INTERNAL
extern "C"
{
// C-style bindings for the reconstruct_solution class
// those will be called from Python ctypes, must match asgard.py

/*!
 * \ingroup asgard_postprocess
 * \brief C binding for reconstruct_solution::reconstruct_solution<double>
 */
void *asgard_make_dreconstruct_solution(int, int64_t, int const[],
                                        int, double const[]);
/*!
 * \ingroup asgard_postprocess
 * \brief C binding for reconstruct_solution::reconstruct_solution<double>
 */
void *asgard_make_freconstruct_solution(int, int64_t, int const[],
                                        int, float const[]);
/*!
 * \ingroup asgard_postprocess
 * \brief C binding that delete the reconstruct_solution object
 */
void asgard_pydelete_reconstruct_solution(void *);
/*!
 * \ingroup asgard_postprocess
 * \brief C binding that delete the reconstruct_solution object and sets pointer to nullptr
 */
void asgard_delete_reconstruct_solution(void **);

/*!
 * \ingroup asgard_postprocess
 * \brief C binding for reconstruct_solution::set_domain_bounds()
 */
void asgard_reconstruct_solution_setbounds(void *, double const[],
                                           double const[]);

/*!
 * \ingroup asgard_postprocess
 * \brief C binding for reconstruct_solution::reconstruct()
 */
void asgard_reconstruct_solution(void const*, double const[], int, double[]);
/*!
 * \ingroup asgard_postprocess
 * \brief C binding for reconstruct_solution::cell_centers()
 */
void asgard_reconstruct_cell_centers(void const *pntr, double x[]);
/*!
 * \ingroup asgard_postprocess
 * \brief C binding to print the library stats
 */
void asgard_print_version_help();
} // extern "C"
#endif
