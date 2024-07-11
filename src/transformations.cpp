#include "transformations.hpp"
#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "batch.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"

namespace asgard
{
//
// set the range specified by first and last,
// starting with the supplied value and incrementing
// that value by stride for each position in the range.
//
template<typename ForwardIterator, typename P>
static void strided_iota(ForwardIterator first, ForwardIterator last, P value,
                         P const stride)
{
  while (first != last)
  {
    *first++ = value;
    value += stride;
  }
}

// perform recursive kronecker product
template<typename P>
fk::vector<P>
kron_d(std::vector<fk::vector<P>> const &operands, int const num_prods)
{
  expect(num_prods > 0);
  if (num_prods == 1)
  {
    return operands[0];
  }
  if (num_prods == 2)
  {
    return operands[0].single_column_kron(operands[1]);
  }
  return kron_d(operands, num_prods - 1)
      .single_column_kron(operands[num_prods - 1]);
}

/* given a vector of matrices, return the Kronecker product of all of them in
 * order */
template<typename P>
fk::matrix<P>
recursive_kron(std::vector<fk::matrix<P, mem_type::view>> &kron_matrices,
               int const index)
{
  expect(index >= 0);
  expect(index < static_cast<int>(kron_matrices.size()));

  if (index == (static_cast<int>(kron_matrices.size()) - 1))
  {
    return fk::matrix<P>(kron_matrices.back());
  }

  else
  {
    return fk::matrix<P>(
        kron_matrices[index].kron(recursive_kron(kron_matrices, index + 1)));
  }
}

template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    PDE<P> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode)
{
  return gen_realspace_transform(pde.get_dimensions(), transformer, quad_mode);
}

/* given a pde, for each dimension create a matrix where the columns are
   legendre basis functions evaluated at the roots */
template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    std::vector<dimension<P>> const &dims,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode)
{
  /* contains a basis matrix for each dimension */
  std::vector<fk::matrix<P>> real_space_transform;
  real_space_transform.reserve(dims.size());

  for (size_t i = 0; i < dims.size(); i++)
  {
    /* get the ith dimension */
    dimension<P> const &d    = dims[i];
    int const level          = d.get_level();
    int const n_segments     = fm::two_raised_to(level);
    int const deg_freedom_1d = d.get_degree() * n_segments;
    P const normalize        = (d.domain_max - d.domain_min) / n_segments;
    /* create matrix of Legendre polynomial basis functions evaluated at the
     * roots */
    auto const roots = legendre_weights<P>(d.get_degree(), -1, 1, quad_mode)[0];
    fk::matrix<P> dimension_transform(roots.size() * n_segments,
                                      deg_freedom_1d);
    /* normalized legendre transformation matrix. Column i is legendre
       polynomial of degree i. element (i, j) is polynomial evaluated at jth
       root of the highest degree polynomial */
    fk::matrix<P> const basis = legendre<P>(roots, d.get_degree())[0] *
                                (static_cast<P>(1.0) / std::sqrt(normalize));
    /* set submatrices of dimension_transform */
    for (int j = 0; j < n_segments; j++)
    {
      int const diagonal_pos = d.get_degree() * j;
      dimension_transform.set_submatrix(roots.size() * j, diagonal_pos, basis);
    }
    real_space_transform.push_back(transformer.apply(dimension_transform, level,
                                                     basis::side::right,
                                                     basis::transpose::trans));
  }
  return real_space_transform;
}

template<typename P>
fk::vector<P>
gen_realspace_nodes(int const degree, int const level, P const min, P const max,
                    quadrature_mode const quad_mode)
{
  int const n      = fm::two_raised_to(level);
  P const h        = (max - min) / n;
  auto const lgwt  = legendre_weights<P>(degree, -1.0, 1.0, quad_mode);
  auto const roots = lgwt[0];

  unsigned int const dof = roots.size();

  // TODO: refactor this whole function.. it does a lot of unnecessary things
  int const mat_dims =
      quad_mode == quadrature_mode::use_degree ? degree * n : dof * n;
  fk::vector<P> nodes(mat_dims);
  for (int i = 0; i < n; i++)
  {
    auto p_val = legendre<P>(roots, degree, legendre_normalization::lin);

    p_val[0] = p_val[0] * sqrt(1.0 / h);

    std::vector<P> xi(dof);
    for (std::size_t j = 0; j < dof; j++)
    {
      xi[j] = (0.5 * (roots(j) + 1.0) + i) * h + min;
    }

    std::vector<int> Iu(dof);
    for (std::size_t j = 0; j < dof; j++)
    {
      Iu[j] = dof * i + j;
    }

    for (std::size_t j = 0; j < dof; j++)
    {
      expect(j <= Iu.size());
      nodes(Iu[j]) = xi[j];
    }
  }

  return nodes;
}

template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    std::vector<dimension_description<P>> const &dims,
    basis::wavelet_transform<P, resource::host> const &transformer,
    quadrature_mode const quad_mode)
{
  /* contains a basis matrix for each dimension */
  std::vector<fk::matrix<P>> real_space_transform;
  real_space_transform.reserve(dims.size());

  for (size_t i = 0; i < dims.size(); i++)
  {
    /* get the ith dimension */
    dimension_description<P> const &d = dims[i];
    int const level                   = d.level;
    int const n_segments              = fm::two_raised_to(level);
    int const deg_freedom_1d          = d.degree * n_segments;
    P const normalize                 = (d.d_max - d.d_min) / n_segments;
    /* create matrix of Legendre polynomial basis functions evaluated at the
     * roots */
    auto const roots = legendre_weights<P>(d.degree, -1, 1, quad_mode)[0];
    fk::matrix<P> dimension_transform(roots.size() * n_segments,
                                      deg_freedom_1d);
    /* normalized legendre transformation matrix. Column i is legendre
       polynomial of degree i. element (i, j) is polynomial evaluated at jth
       root of the highest degree polynomial */
    fk::matrix<P> const basis = legendre<P>(roots, d.degree)[0] *
                                (static_cast<P>(1.0) / std::sqrt(normalize));
    /* set submatrices of dimension_transform */
    for (int j = 0; j < n_segments; j++)
    {
      int const diagonal_pos = d.degree * j;
      dimension_transform.set_submatrix(roots.size() * j, diagonal_pos, basis);
    }
    real_space_transform.push_back(transformer.apply(dimension_transform, level,
                                                     basis::side::right,
                                                     basis::transpose::trans));
  }
  return real_space_transform;
}

template<typename P>
void wavelet_to_realspace(
    PDE<P> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space, quadrature_mode const quad_mode)
{
  tools::time_event performance("wavelet_to_realspace");
  wavelet_to_realspace(pde.get_dimensions(), wave_space, table, transformer,
                       workspace, real_space, quad_mode);
}

template<typename P>
void wavelet_to_realspace(
    std::vector<dimension<P>> const &dims, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space, quadrature_mode const quad_mode)
{
  std::vector<batch_chain<P, resource::host>> chain;

  /* generate the wavelet-to-real-space transformation matrices for each
   * dimension */
  std::vector<fk::matrix<P>> real_space_transform =
      gen_realspace_transform(dims, transformer, quad_mode);

  // FIXME Assume the degree in the first dimension is equal across all the
  // remaining dimensions
  auto const stride = std::pow(dims[0].get_degree(), dims.size());

  fk::vector<P, mem_type::owner, resource::host> accumulator(real_space.size());
  fk::vector<P, mem_type::view, resource::host> real_space_accumulator(
      accumulator);

  for (int64_t i = 0; i < table.size(); i++)
  {
    std::vector<fk::matrix<P, mem_type::const_view>> kron_matrices;
    kron_matrices.reserve(dims.size());
    auto const coords = table.get_coords(i);

    for (size_t j = 0; j < dims.size(); j++)
    {
      auto const id =
          elements::get_1d_index(coords(j), coords(j + dims.size()));
      auto const degree = dims[j].get_degree();
      fk::matrix<P, mem_type::const_view> sub_matrix(
          real_space_transform[j], 0, real_space_transform[j].nrows() - 1,
          id * degree, (id + 1) * degree - 1);
      kron_matrices.push_back(sub_matrix);
    }

    /* create a view of a section of the wave space vector */
    fk::vector<P, mem_type::const_view> const x(wave_space, i * stride,
                                                (i + 1) * stride - 1);

    chain.emplace_back(kron_matrices, x, workspace, real_space_accumulator);
  }

  /* clear out the vector */
  real_space.scale(0);

  for (auto const &link : chain)
  {
    link.execute();
    real_space = real_space + real_space_accumulator;
  }
}

template<typename P>
void wavelet_to_realspace(
    std::vector<dimension_description<P>> const &dims,
    fk::vector<P> const &wave_space, elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space, quadrature_mode const quad_mode)
{
  std::vector<batch_chain<P, resource::host>> chain;

  /* generate the wavelet-to-real-space transformation matrices for each
   * dimension */
  std::vector<fk::matrix<P>> real_space_transform =
      gen_realspace_transform(dims, transformer, quad_mode);

  // FIXME Assume the degree in the first dimension is equal across all the
  // remaining dimensions
  auto const stride = std::pow(dims[0].degree, dims.size());

  fk::vector<P, mem_type::owner, resource::host> accumulator(real_space.size());
  fk::vector<P, mem_type::view, resource::host> real_space_accumulator(
      accumulator);

  for (int64_t i = 0; i < table.size(); i++)
  {
    std::vector<fk::matrix<P, mem_type::const_view>> kron_matrices;
    kron_matrices.reserve(dims.size());
    auto const coords = table.get_coords(i);

    for (size_t j = 0; j < dims.size(); j++)
    {
      auto const id =
          elements::get_1d_index(coords(j), coords(j + dims.size()));
      auto const degree = dims[j].degree;
      fk::matrix<P, mem_type::const_view> sub_matrix(
          real_space_transform[j], 0, real_space_transform[j].nrows() - 1,
          id * degree, (id + 1) * degree - 1);
      kron_matrices.push_back(sub_matrix);
    }

    /* create a view of a section of the wave space vector */
    fk::vector<P, mem_type::const_view> const x(wave_space, i * stride,
                                                (i + 1) * stride - 1);

    chain.emplace_back(kron_matrices, x, workspace, real_space_accumulator);
  }

  /* clear out the vector */
  real_space.scale(0);

  for (auto const &link : chain)
  {
    link.execute();
    real_space = real_space + real_space_accumulator;
  }
}

template<typename P>
void combine_dimensions(int const degree, elements::table const &table,
                        int const start_element, int const stop_element,
                        std::vector<fk::vector<P>> const &vectors,
                        P const time_scale,
                        fk::vector<P, mem_type::view> result)
{
  int const num_dims = vectors.size();
  expect(num_dims > 0);
  expect(start_element >= 0);
  expect(stop_element >= start_element);
  expect(stop_element < table.size());

  int64_t const vector_size =
      (stop_element - start_element + 1) * std::pow(degree, num_dims);

  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  expect(vector_size < INT_MAX);
  expect(result.size() == vector_size);

  for (int i = start_element; i <= stop_element; ++i)
  {
    std::vector<fk::vector<P>> kron_list;
    fk::vector<int> const coords = table.get_coords(i);
    for (int j = 0; j < num_dims; ++j)
    {
      // iterating over cell coords;
      // first num_dims entries in coords are level coords
      int const id = elements::get_1d_index(coords(j), coords(j + num_dims));
      int const index_start = id * degree;
      // index_start and index_end describe a subvector of length degree;
      // for deg = 1, this is a vector of one element
      int const index_end =
          degree > 1 ? (((id + 1) * degree) - 1) : index_start;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    int const start_index = (i - start_element) * std::pow(degree, num_dims);
    int const stop_index  = start_index + std::pow(degree, num_dims) - 1;

    // call kron_d and put the output in the right place of the result
    fk::vector<P, mem_type::view>(result, start_index, stop_index) =
        kron_d(kron_list, kron_list.size()) * time_scale;
  }
}

// FIXME this function will need to change once dimensions can have different
// degree...
// combine components and create the portion of the multi-d vector associated
// with the provided start and stop element bounds (inclusive)
template<typename P>
fk::vector<P>
combine_dimensions(int const degree, elements::table const &table,
                   int const start_element, int const stop_element,
                   std::vector<fk::vector<P>> const &vectors,
                   P const time_scale)
{
  int64_t const vector_size =
      (stop_element - start_element + 1) * std::pow(degree, vectors.size());

  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  expect(vector_size < INT_MAX);
  fk::vector<P> combined(vector_size);

  combine_dimensions(degree, table, start_element, stop_element, vectors,
                     time_scale, fk::vector<P, mem_type::view>(combined));

  return combined;
}

template<typename P>
fk::vector<P> sum_separable_funcs(
    std::vector<md_func_type<P>> const &funcs,
    std::vector<dimension<P>> const &dims,
    adapt::distributed_grid<P> const &grid,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const degree, P const time)
{
  auto const my_subgrid = grid.get_subgrid(get_rank());
  // FIXME assume uniform degree
  auto const dof = std::pow(degree, dims.size()) * my_subgrid.nrows();
  fk::vector<P> combined(dof);
  for (auto const &md_func : funcs)
  {
    expect(md_func.size() >= dims.size());

    // calculate the time multiplier if there is an extra function for time
    // TODO: this is a hack to append a time function.. md_func_type should be a
    // struct since this last function is technically a scalar_func
    bool has_time_func      = md_func.size() == dims.size() + 1 ? true : false;
    P const time_multiplier = has_time_func
                                  ? md_func.back()(fk::vector<P>(), time)[0]
                                  : static_cast<P>(1.0);
    auto const func_vect    = transform_and_combine_dimensions(
        dims, md_func, grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop, degree, time, time_multiplier);
    fm::axpy(func_vect, combined);
  }
  return combined;
}

#ifdef ASGARD_ENABLE_DOUBLE
template fk::matrix<double>
recursive_kron(std::vector<fk::matrix<double, mem_type::view>> &kron_matrices,
               int const index);
template std::vector<fk::matrix<double>> gen_realspace_transform(
    PDE<double> const &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template fk::vector<double>
gen_realspace_nodes(int const degree, int const level, double const min,
                    double const max, quadrature_mode const quad_mode);
template std::vector<fk::matrix<double>> gen_realspace_transform(
    std::vector<dimension<double>> const &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template std::vector<fk::matrix<double>> gen_realspace_transform(
    std::vector<dimension_description<double>> const &pde,
    basis::wavelet_transform<double, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    PDE<double> const &pde, fk::vector<double> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<double, resource::host> const &transformer,
    std::array<fk::vector<double, mem_type::view, resource::host>, 2>
        &workspace,
    fk::vector<double> &real_space, quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    std::vector<dimension<double>> const &pde,
    fk::vector<double> const &wave_space, elements::table const &table,
    basis::wavelet_transform<double, resource::host> const &transformer,
    std::array<fk::vector<double, mem_type::view, resource::host>, 2>
        &workspace,
    fk::vector<double> &real_space, quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    std::vector<dimension_description<double>> const &pde,
    fk::vector<double> const &wave_space, elements::table const &table,
    basis::wavelet_transform<double, resource::host> const &transformer,
    std::array<fk::vector<double, mem_type::view, resource::host>, 2>
        &workspace,
    fk::vector<double> &real_space, quadrature_mode const quad_mode);
template fk::vector<double>
combine_dimensions(int const, elements::table const &, int const, int const,
                   std::vector<fk::vector<double>> const &, double const = 1.0);
template void
combine_dimensions<double>(int const, elements::table const &, int const,
                           int const, std::vector<fk::vector<double>> const &,
                           double const, fk::vector<double, mem_type::view>);
template fk::vector<double> sum_separable_funcs(
    std::vector<md_func_type<double>> const &funcs,
    std::vector<dimension<double>> const &dims,
    adapt::distributed_grid<double> const &grid,
    basis::wavelet_transform<double, resource::host> const &transformer,
    int const degree, double const time);
#endif

#ifdef ASGARD_ENABLE_FLOAT
template fk::matrix<float>
recursive_kron(std::vector<fk::matrix<float, mem_type::view>> &kron_matrices,
               int const index);
template std::vector<fk::matrix<float>> gen_realspace_transform(
    PDE<float> const &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template fk::vector<float>
gen_realspace_nodes(int const degree, int const level, float const min,
                    float const max, quadrature_mode const quad_mode);
template std::vector<fk::matrix<float>> gen_realspace_transform(
    std::vector<dimension<float>> const &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template std::vector<fk::matrix<float>> gen_realspace_transform(
    std::vector<dimension_description<float>> const &pde,
    basis::wavelet_transform<float, resource::host> const &transformer,
    quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    PDE<float> const &pde, fk::vector<float> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<float, resource::host> const &transformer,
    std::array<fk::vector<float, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<float> &real_space, quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    std::vector<dimension<float>> const &pde,
    fk::vector<float> const &wave_space, elements::table const &table,
    basis::wavelet_transform<float, resource::host> const &transformer,
    std::array<fk::vector<float, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<float> &real_space, quadrature_mode const quad_mode);
template void wavelet_to_realspace(
    std::vector<dimension_description<float>> const &pde,
    fk::vector<float> const &wave_space, elements::table const &table,
    basis::wavelet_transform<float, resource::host> const &transformer,
    std::array<fk::vector<float, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<float> &real_space, quadrature_mode const quad_mode);
template fk::vector<float>
combine_dimensions(int const, elements::table const &, int const, int const,
                   std::vector<fk::vector<float>> const &, float const = 1.0);
template void
combine_dimensions<float>(int const, elements::table const &, int const,
                          int const, std::vector<fk::vector<float>> const &,
                          float const, fk::vector<float, mem_type::view>);
template fk::vector<float> sum_separable_funcs(
    std::vector<md_func_type<float>> const &funcs,
    std::vector<dimension<float>> const &dims,
    adapt::distributed_grid<float> const &grid,
    basis::wavelet_transform<float, resource::host> const &transformer,
    int const degree, float const time);
#endif

} // namespace asgard
