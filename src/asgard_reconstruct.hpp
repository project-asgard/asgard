#pragma once

#include "asgard_interpolation.hpp"

namespace asgard
{

/*!
 * \brief Reconstruct the solution at arbitrary points in the domain from wavelet data
 *
 * The class is inteded to be used from within Python for plotting
 * and post-processing of the data. The API uses raw-arrays to facilitate
 * interfacing with the Python ctypes module.
 *
 * The reconstruction algorihms are taken from Tasmanian.
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

protected:
  /*!
   * \brief Prepare the inernal data-structures for fast reconstruction
   *
   * The way things are commonly done in ASGarD is to build the basis on a full
   * grid (at least in 1D), then move to wavelet space using a matrix product.
   * The process is expensive and becomes less viable after level 10.
   *
   * The alternative is to find analytic expressions for the wavelets and avoid
   * going to "real" or "non-wavelet" space. This leads to limitations on
   * the polynomial degree, i.e., wavelets have to be hard-coded (for now).
   *
   * Furthermore, since the basis is local, many of the functions can be
   * skipped in the reconstruction. The sparse grids cells natually form
   * a directed-asyclic-graph (DAG) and we can use the fact that the support
   * of the children is included in the support of the parent, so if the parent
   * is not supported over a given point in the domain, all children are
   * also non-supported and can be skipepd.
   *
   * On the other hand, since the DAG allows for multiple parents, it is likely
   * to double-count the supported functions. This method repalces the DAG with
   * a tree structure by removing redundant connections. The main goals it that
   * all supported functions can be identified easily. There may be multiple
   * trees due to orphan nodes in the refinement process.
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
   * returns the value of the sum of basis times coefficients,
   * the optional is empty if the basis is not supported.
   */
  std::optional<double>
  basis_value(int const p[], double const x[], double const c[]) const;
  //! Evaluate the loaded approximation at point x
  double walk_tree(const double x[]) const;

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
};

} // namespace asgard

extern "C"
{
// C-style bindings for the reconstruct_solution class
// those will be called from Python ctypes, must match asgard.py
void *asgard_make_dreconstruct_solution(int, int64_t, int const[],
                                        int, double const[]);
void *asgard_make_freconstruct_solution(int, int64_t, int const[],
                                        int, float const[]);
void asgard_pydelete_reconstruct_solution(void *);
void asgard_delete_reconstruct_solution(void **);

void asgard_reconstruct_solution_setbounds(void *, double const[],
                                           double const[]);

void asgard_reconstruct_solution(void *, double const[], int, double[]);

} // extern "C"
