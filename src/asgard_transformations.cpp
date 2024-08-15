#include "asgard_transformations.hpp"

#include "asgard_small_mats.hpp"

namespace asgard
{
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
    int const deg_freedom_1d = (d.get_degree() + 1) * n_segments;
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
      int const diagonal_pos = (d.get_degree() + 1) * j;
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
      quad_mode == quadrature_mode::use_degree ? (degree + 1) * n : dof * n;
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

  int64_t const stride = fm::ipow(dims[0].get_degree() + 1, dims.size());

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
          id * (degree + 1), (id + 1) * (degree + 1) - 1);
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

  int const pblock = degree + 1;

  int64_t const vector_size =
      (stop_element - start_element + 1) * fm::ipow(pblock, num_dims);

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
      int const index_start = id * pblock;
      // index_start and index_end describe a subvector of length degree + 1;
      // for deg = 1, this is a vector of one element
      int const index_end =
          degree > 0 ? (((id + 1) * pblock) - 1) : index_start;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    int const start_index = (i - start_element) * fm::ipow(pblock, num_dims);
    int const stop_index  = start_index + fm::ipow(pblock, num_dims) - 1;

    // call kron_d and put the output in the right place of the result
    fk::vector<P, mem_type::view>(result, start_index, stop_index) =
        kron_d(kron_list, kron_list.size()) * time_scale;
  }
}

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
      (stop_element - start_element + 1) * fm::ipow(degree + 1, vectors.size());

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
  int64_t const dof = fm::ipow(degree + 1, dims.size()) * my_subgrid.nrows();
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

template<typename P>
void hierarchy_manipulator<P>::make_mass(int dim, int level, mass_list &mass) const
{
  int const num_cells = fm::ipow2(level);
  int const num_quad  = leg_unscal.stride();
  int const pdof      = degree_ + 1;
  int const bsize     = pdof * pdof;

  std::vector<P> mat(int64_t(pdof) * pdof * num_cells);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    // void gemm3(int const &n, int const &m, P const A[], P const d[], P const B[], P C[])
    smmat::gemm3(pdof, num_quad, leg_vals[0], quad_dv[dim].data() + i * num_quad,
                 leg_unscal[0], mat.data() + i * bsize);
  }

  switch (degree_)
  {
  case 0:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      mat[i] = P{1} / mat[i];
    break;
  case 1:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      smmat::inv2by2(mat.data() + i * bsize);
    break;
  default:
#pragma omp parallel for
    for (int i = 0; i < num_cells; i++)
      smmat::potrf(pdof, mat.data() + i * bsize);
    break;
  }

  mass[dim]->set(level, std::move(mat));
}

template<typename P>
void hierarchy_manipulator<P>::project1d(int d, int level, P const dsize, P const *mass) const
{
  int const num_cells = fm::ipow2(level);

  int const num_quad = quad.stride();
  int const pdof     = degree_ + 1;

  expect(fvals.size() == static_cast<size_t>(num_cells * num_quad));

  stage0.resize(pdof * num_cells);

  P const scale = 0.5 * std::pow(is2, level - 1) * std::sqrt(dsize);

#pragma omp parallel for
  for (int i = 0; i < num_cells; i++)
  {
    smmat::gemv(pdof, num_quad, leg_vals[0], &fvals[i * num_quad],
                &stage0[i * pdof]);
    smmat::scal(pdof, scale, &stage0[i * pdof]);
  }

  if (mass != nullptr)
  {
    switch (degree_)
    {
    case 0:
#pragma omp parallel for
      for (int i = 0; i < num_cells; i++)
        stage0[i] *= mass[i];
      break;
    case 1:
#pragma omp parallel for
      for (int i = 0; i < num_cells; i++)
        smmat::gemv2by2(mass + 4 * i, stage0.data() + 2 * i);
      break;
    default:
#pragma omp parallel for
      for (int i = 0; i < num_cells; i++)
        smmat::posv(pdof, mass + i * pdof * pdof, stage0.data() + i * pdof);
      break;
    };
  }

  pf[d].resize(pdof * num_cells);

  // stage0 contains the projection data per-cell
  // pf has the correct size to take the data, so project all levels up
  switch (degree_)
  { // hardcoded degrees first, the default uses the projection matrices
  case 0:
    projectlevels<0>(d, level);
    break;
  case 1:
    projectlevels<1>(d, level);
    break;
  default:
    projectlevels<-1>(d, level);
  };
}

template<typename P>
template<int degree>
void hierarchy_manipulator<P>::projectup2(P const *raw, P *fin) const
{
  if constexpr (degree == 0)
  {
    P constexpr s22 = 0.5 * s2;
    fin[0] = s22 * raw[0] + s22 * raw[1];
    fin[1] = -s22 * raw[0] + s22 * raw[1];
  }
  else if constexpr (degree == 1)
  {
    P constexpr is2h = 0.5 * is2;
    P constexpr is64  = s6 / 4.0;

    fin[0] = is2 * raw[0]                   + is2 * raw[2];
    fin[1] = -is64 * raw[0] + is2h * raw[1] + is64 * raw[2] + is2h * raw[3];
    fin[2] = -is2 * raw[1] + is2 * raw[3];
    fin[3] = is2h * raw[0] + is64 * raw[1] - is2h * raw[2] + is64 * raw[3];
  }
  else
  {
    int const n = 2 * (degree_ + 1);
    smmat::gemv(n, n, pmats.data(), raw, fin);
  }
}

template<typename P>
template<int degree>
void hierarchy_manipulator<P>::projectup(int num_final, P const *raw, P *upper, P *fin) const
{
  if constexpr (degree == 0)
  {
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      P constexpr s22 = 0.5 * s2;
      P const r0 = raw[2 * i];
      P const r1 = raw[2 * i + 1];
      upper[i] = s22 * r0 + s22 * r1;
      fin[i]   = -s22 * r0 + s22 * r1;
    }
  }
  else if constexpr (degree == 1)
  {
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      P constexpr is2h = 0.5 * is2;
      P constexpr is64  = s6 / 4.0;
      P const r0 = raw[4 * i];
      P const r1 = raw[4 * i + 1];
      P const r2 = raw[4 * i + 2];
      P const r3 = raw[4 * i + 3];
      upper[2 * i]     = is2 * r0                   + is2 * r2;
      upper[2 * i + 1] = -is64 * r0 + is2h * r1 + is64 * r2 + is2h * r3;
      fin[2 * i]       = -is2 * r1 + is2 * r3;
      fin[2 * i + 1]   = is2h * r0 + is64 * r1 - is2h * r2 + is64 * r3;
    }
  }
  else
  {
    int const pdof = degree_ + 1;
#pragma omp parallel for
    for (int i = 0; i < num_final; i++)
    {
      smmat::gemv(pdof, 2 * pdof, pmatup, &raw[2 * pdof * i], &upper[i * pdof]);
      smmat::gemv(pdof, 2 * pdof, pmatlev, &raw[2 * pdof * i], &fin[i * pdof]);
    }
  }
}

template<typename P>
template<int degree>
void hierarchy_manipulator<P>::projectlevels(int d, int level) const
{
  switch (level)
  {
  case 0:
    std::copy(stage0.begin(), stage0.end(), pf[d].begin()); // nothing to project upwards
    break;
  case 1:
    projectup2<degree>(stage0.data(), pf[d].data()); // level 0 and 1
    break;
  default: {
      stage1.resize(stage0.size() / 2);
      int const pdof = degree_ + 1;

      P *w0  = stage0.data();
      P *w1  = stage1.data();
      int num = static_cast<int>(pf[d].size() / (2 * pdof));
      P *fin = pf[d].data() + num * pdof;
      for (int l = level; l > 1; l--)
      {
        projectup<degree>(num, w0, w1, fin);
        std::swap(w0, w1);
        num /= 2;
        fin -= num * pdof;
      }
      projectup2<degree>(w0, pf[d].data());
    }
  }
}

template<typename P>
void hierarchy_manipulator<P>::prepare_quadrature(int d, int num_cells) const
{
  int const num_quad = quad.stride();

  // if quadrature is already set for the correct level, no need to do anything
  // this assumes that the min/max of the domain does not change
  if (quad_points[d].size() == static_cast<size_t>(num_quad * num_cells))
    return;

  quad_points[d].resize(num_quad * num_cells);

  P const cell_size = (dmax[d] - dmin[d]) / P(num_cells);

  P mid       = dmin[d] + 0.5 * cell_size;
  P const slp = 0.5 * cell_size;

  P *iq = quad_points[d].data();
  for (int i : indexof<int>(num_cells))
  {
    ignore(i);
    for (int j : indexof<int>(num_quad))
      iq[j] = slp * quad[points][j] + mid;
    mid += cell_size;
    iq  += num_quad;
  }
}

template<typename P>
void hierarchy_manipulator<P>::setup_projection_matrices()
{
    int const num_quad = quad.stride();

  // leg_vals is a small matrix with the values of Legendre polynomials
  // scaled by the quadrature weights
  // the final structure is such that small matrix leg_vals times the
  // vector of f(x_i) at quadrature points x_i will give us the projection
  // of f onto the Legendre polynomial basis
  // scaled by the l-2 volume of the cell, this is the local projection of f(x)
  // leg_unscal is the transpose of leg_vals and unscaled by the quadrature w.
  // if rho(x_i) are local values of the mass weight, the local mass matrix is
  // leg_vals * diag(rho(x_i)) * leg_unscal
  leg_vals   = vector2d<P>(degree_ + 1, num_quad);
  leg_unscal = vector2d<P>(num_quad, degree_ + 1);

  P const *qpoints = quad[points];
  // using the recurrence: L_n = ((2n - 1) L_{n-1} - (n - 1) L_{n-2}) / n
  for (int i : indexof<int>(num_quad))
  {
    P *l = leg_vals[i];
    l[0] = 1.0;
    if (degree_ > 0)
      l[1] = qpoints[i];
    for (int j = 2; j <= degree_; j++)
      l[j] = ((2 * j - 1) * qpoints[i] * l[j-1] - (j - 1) * l[j-2]) / P(j);
  }

  for (int j = 0; j <= degree_; j++)
  {
    P const scale = std::sqrt( (2 * j + 1) / P(2) );
    for (int i : indexof<int>(num_quad))
      leg_unscal[j][i] = scale * leg_vals[i][j];
    for (int i : indexof<int>(num_quad))
      leg_vals[i][j] *= scale * quad[weights][i];
  }

  if (degree_ >= 2) // need projection matrices, degree_ <= 1 are hard-coded
  {
    auto rawmats = generate_multi_wavelets<P>(degree_);
    int const pdof = degree_ + 1;
    // copy the matrices twice, once for level 1->0 and once for generic levels
    pmats.resize(8 * pdof * pdof);
    auto ip = pmats.data();
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[0].data(0, i), pdof, ip);
      ip = std::copy_n(rawmats[2].data(0, i), pdof, ip);
    }
    for (int i : indexof<int>(pdof)) {
      ip = std::copy_n(rawmats[1].data(0, i), pdof, ip);
      ip = std::copy_n(rawmats[3].data(0, i), pdof, ip);
    }

    pmatup = ip;
    pmatlev = pmatup + 2 * pdof * pdof;

    for (int j = 0; j < 4; j++)
      for (int i : indexof<int>(pdof))
        ip = std::copy_n(rawmats[j].data(0, i), pdof, ip);
  }
}

#ifdef ASGARD_ENABLE_DOUBLE
template class hierarchy_manipulator<double>;

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
template class hierarchy_manipulator<float>;

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
