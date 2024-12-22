#include "asgard_moment.hpp"
#include "asgard_small_mats.hpp"

namespace asgard
{

template<typename P>
moments1d<P>::moments1d(int num_mom, int degree, int max_level,
                        std::vector<dimension<P>> const &dims)
  : num_mom_(num_mom), num_dims_(static_cast<int>(dims.size())),
    degree_(degree)
{
  expect(num_dims_ > 1); // cannot have moments for 1D
  expect(num_dims_ <= 4); // cannot have more than 3 velocity dims
  P constexpr s2 = 1.41421356237309505; // sqrt(2.0)

  std::array<P, max_num_dimensions> dim_scale;
  for (int d : iindexof(num_dims_))
    dim_scale[d] = dims[d].domain_max - dims[d].domain_min;

  // Legendre and wavelet polynomials
  vector2d<P> pleg = basis::legendre_poly<P>(degree_);
  vector2d<P> pwav = basis::wavelet_poly(pleg, degree_);

  int const pdof   = degree_ + 1;
  int const nblock = num_mom_ * pdof;
  int const nump   = fm::ipow2(max_level);

  for (int d : iindexof(num_dims_))
  {
    if (d == 0)
      continue; // first dimension is the position

    integ[d] = vector2d<P>(nblock, nump);

    P const amin = dims[d].domain_min;
    P const amax = dims[d].domain_max;

    // global scale factor, the basis is unit normalized
    P const scale = 1.0 / (s2 * std::sqrt(amax - amin));

#pragma omp parallel
    {
      basis::canonical_integrator quad(num_mom_, degree_);
      std::vector<P> work(4 * quad.left_nodes().size());

#pragma omp for
      for (int i = 1; i < nump; i++)
      {
        int const level  = fm::intlog2(i);   // previous level of i
        int const istart = fm::ipow2(level); // first index on this level

        P const dx = (amax - amin) / istart; // cell size
        P const a  = amin + (i - istart) * dx;

        span2d<P> block(pdof, num_mom_, integ[d][i]);
        integrate(quad, a, a + dx, dims[d].volume_jacobian_dV, pwav, work, block);

        P s = ((level > 1) ? fm::powi<P>(s2, level -1) : P{1}) * scale;
        smmat::scal(pdof * num_mom_, s, integ[d][i]);
      }

#pragma omp single
      {
        integrate(quad, amin, amax, dims[d].volume_jacobian_dV, pleg, work,
                  span2d<P>(pdof, num_mom_, integ[d][0]));
        smmat::scal(pdof * num_mom_, scale, integ[d][0]);
      }
    }
  }
}

template<typename P>
void moments1d<P>::integrate(
    basis::canonical_integrator const &quad, P a, P b, g_func_type<P> const &dv,
    vector2d<P> const &basis, std::vector<P> &work, span2d<P> integ) const
{
  expect(work.size() == 4 * quad.left_nodes().size());
  expect(basis.num_strips() == degree_ + 1);
  expect(integ.stride() == degree_ + 1);
  expect(integ.num_strips() == num_mom_);

  size_t nquad = quad.left_nodes().size(); // num-quad-points

  // holds the values of the moment-weight, e.g., 1, v, v^2 ...
  P *ml = work.data();
  P *mr = ml + nquad;
  // holds the values of the v-nodes, e.g., v_1, v_2 ...
  P *nl = mr + nquad;
  P *nr = nl + nquad;

  std::copy_n(quad.left_nodes().begin(), nquad, nl);
  std::copy_n(quad.right_nodes().begin(), nquad, nr);

  P const scal = b - a; // domain scale
  { // convert the canonical interval to (b, a)
    P slope = 0.5 * scal, intercept = 0.5 * (b + a);
    for (int i : iindexof(2 * nquad))
      nl[i] = slope * nl[i] + intercept;
  }

  // setting the zeroth moment
  if (dv) // if using non-Cartesian coords
    for (int i : iindexof(2 * nquad))
      ml[i] = dv(nl[i], 0);
  else
    std::fill_n(ml, 2 * nquad, 1.0);

  for (int moment : iindexof(num_mom_))
  {
    if (moment > 0)
      for (int i : iindexof(2 * nquad)) // does both left/right parts
        ml[i] *= nl[i];

    P *ii = integ[moment];
    if (basis.stride() == degree_ + 1)
      for (int d : iindexof(degree_ + 1))
        *ii++ = scal * quad.integrate_lmom(ml, mr, basis[d]);
    else
      for (int d : iindexof(degree_ + 1))
        *ii++ = scal * quad.integrate_wmom(ml, mr, basis[d]);
  }
}

template<typename P>
void moments1d<P>::project_moments(
    int const dim0_level, std::vector<P> const &state,
    elements::table const &etable, std::vector<P> &moments) const
{
  tools::time_event performance("moments project");

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(dim0_level);
  if (moments.empty())
    moments.resize(nout * num_mom_ * pdof);
  else {
    moments.resize(nout * num_mom_ * pdof);
    std::fill(moments.begin(), moments.end(), P{0});
  }

  vector2d<int> cells = etable.get_cells();

  auto const ncells = cells.num_strips();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(num_mom_ * pdof, nout, moments.data());

  std::vector<P> work; // persistant workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<2>(x[i], idx, span2d<P>(pdof, num_mom_, smom[idx[0]]), work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<3>(x[i], idx, span2d<P>(pdof, num_mom_, smom[idx[0]]), work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<4>(x[i], idx, span2d<P>(pdof, num_mom_, smom[idx[0]]), work);
      }
      break;
  }
}

template<typename P>
template<int ndims>
void moments1d<P>::project_cell(P const x[], int const idx[], span2d<P> moments,
                                std::vector<P> &work) const
{
  int const pdof = degree_ + 1;
  if constexpr (ndims == 2) // reducing only one dimension
  {
    for (int m : iindexof(num_mom_))
    {
      P const *wm = integ[1][idx[1]] + m * pdof; // moment weights
      P *mout = moments[m];
      for (int i = 0; i < pdof; i++)
      {
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * x[i * pdof + j];
      }
    }
  }
  else if constexpr (ndims == 3) // reducing 2 dimensions, using work as temp storage
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof));
    for (int m : iindexof(num_mom_))
    {
      P const *wm = integ[2][idx[2]] + m * pdof; // moment weights

      for (int i = 0; i < pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[1][idx[1]] + m * pdof;
      P *mout = moments[m];
      for (int i = 0; i < pdof; i++)
      {
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * work[i * pdof + j];
      }
    }
  }
  else if constexpr (ndims == 4) // reducing 3 dims, using work in 2 stages
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof * pdof + pdof * pdof));
    for (int m : iindexof(num_mom_))
    {
      P const *wm = integ[3][idx[3]] + m * pdof; // moment weights

      for (int i = 0; i < pdof * pdof * pdof; i++)
      {
        work[i] = 0;
        for (int j = 0; j < pdof; j++)
          work[i] += wm[j] * x[i * pdof + j];
      }

      wm = integ[2][idx[2]] + m * pdof;
      P *t = work.data() + pdof * pdof * pdof;

      for (int i = 0; i < pdof * pdof; i++)
      {
        t[i] = 0;
        for (int j = 0; j < pdof; j++)
          t[i] += wm[j] * work[i * pdof + j];
      }

      wm = integ[1][idx[1]] + m * pdof;
      P *mout = moments[m];
      for (int i = 0; i < pdof; i++)
      {
        for (int j = 0; j < pdof; j++)
          mout[i] += wm[j] * t[i * pdof + j];
      }
    }
  }
}

template<typename P>
void moments1d<P>::project_moment(
    int const mom, int const dim0_level, std::vector<P> const &state,
    elements::table const &etable, std::vector<P> &moment) const
{
  tools::time_event performance("moment project");

  int const pdof = degree_ + 1;
  int const nout = fm::ipow2(dim0_level);
  if (moment.empty())
    moment.resize(nout * pdof);
  else {
    moment.resize(nout * pdof);
    std::fill(moment.begin(), moment.end(), P{0});
  }

  vector2d<int> cells = etable.get_cells();

  auto const ncells = cells.num_strips();

  int64_t const tsize = fm::ipow(pdof, num_dims_);

  span2d<P const> x(tsize, ncells, state.data());

  span2d<P> smom(pdof, nout, moment.data());

  std::vector<P> work; // persistant workspace

  switch (num_dims_) {
    case 2:
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<2>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 3:
      work.resize(pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<3>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
    case 4:
      work.resize(pdof * pdof * pdof + pdof * pdof);
      for (int64_t i = 0; i < ncells; i++)
      {
        int const *idx = cells[i];
        project_cell<4>(mom, x[i], idx, smom[idx[0]], work);
      }
      break;
  }
}

template<typename P>
template<int ndims>
void moments1d<P>::project_cell(
    int const mom, P const x[], int const idx[], P moment[], std::vector<P> &work) const
{
  int const pdof = degree_ + 1;
  if constexpr (ndims == 2) // reducing only one dimension
  {
    P const *wm = integ[1][idx[1]] + mom * pdof; // moment weights
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * x[i * pdof + j];
    }
  }
  else if constexpr (ndims == 3) // reducing 2 dimensions, using work as temp storage
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof));
    P const *wm = integ[2][idx[2]] + mom * pdof; // moment weights

    for (int i = 0; i < pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[1][idx[1]] + mom * pdof;
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * work[i * pdof + j];
    }
  }
  else if constexpr (ndims == 4) // reducing 3 dims, using work in 2 stages
  {
    expect(work.size() == static_cast<size_t>(pdof * pdof * pdof + pdof * pdof));
    P const *wm = integ[3][idx[3]] + mom * pdof; // moment weights

    for (int i = 0; i < pdof * pdof * pdof; i++)
    {
      work[i] = 0;
      for (int j = 0; j < pdof; j++)
        work[i] += wm[j] * x[i * pdof + j];
    }

    wm = integ[2][idx[2]] + mom * pdof;
    P *t = work.data() + pdof * pdof * pdof;

    for (int i = 0; i < pdof * pdof; i++)
    {
      t[i] = 0;
      for (int j = 0; j < pdof; j++)
        t[i] += wm[j] * work[i * pdof + j];
    }

    wm = integ[1][idx[1]] + mom * pdof;
    for (int i = 0; i < pdof; i++)
    {
      for (int j = 0; j < pdof; j++)
        moment[i] += wm[j] * t[i * pdof + j];
    }
  }
}

template<typename P>
moment<P>::moment(std::vector<md_func_type<P>> md_funcs_)
    : md_funcs(std::move(md_funcs_))
{}

// Creates the coefficients of the moment vector on each domain.
// No mass matrix inversion is done.
template<typename P>
void moment<P>::createFlist(PDE<P> const &pde)
{
  std::size_t num_md_funcs = this->md_funcs.size();

  auto const &dims     = pde.get_dimensions();
  std::size_t num_dims = dims.size();

  this->fList.clear();
  this->fList.resize(num_md_funcs);

  basis::wavelet_transform<P, resource::host> const transformer(pde);

  for (std::size_t s = 0; s < num_md_funcs; ++s)
  {
    auto const &md_func = this->md_funcs[s];
    for (std::size_t d = 0; d < num_dims; ++d)
    {
      fList[s].push_back(forward_transform<P>(
          dims[d], md_func[d], dims[d].volume_jacobian_dV, transformer));
    }
  }
}

// Actually contstructs the moment vector using fList.
// Calculate only if adapt is true or the vector field is empty
template<typename P>
void moment<P>::createMomentVector(PDE<P> const &pde,
                                   elements::table const &hash_table)
{
  // check that fList has been constructed
  expect(this->fList.size() > 0);

  if (this->vector.empty() or pde.options().adapt_threshold)
  {
    distribution_plan const plan = get_plan(get_num_ranks(), hash_table);
    auto rank                    = get_rank();
    int const degree             = pde.get_dimensions()[0].get_degree();
    auto tmp = combine_dimensions(degree, hash_table, plan.at(rank).row_start,
                                  plan.at(rank).row_stop, this->fList[0]);
    this->vector      = std::move(tmp);
    auto num_md_funcs = md_funcs.size();
    for (std::size_t s = 1; s < num_md_funcs; ++s)
    {
      tmp = combine_dimensions(degree, hash_table, plan.at(rank).row_start,
                               plan.at(rank).row_stop, this->fList[s]);
      std::transform(tmp.begin(), tmp.end(), this->vector.begin(),
                     this->vector.begin(), std::plus<>{});
    }
  }
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims; ++d)
  {
    indices(d) = coords(d) * (degree + 1);
  }
  return indices;
}

template<typename P>
void moment<P>::createMomentReducedMatrix(PDE<P> const &pde,
                                          elements::table const &hash_table)
{
  tools::time_event performance("createMomentMatrix");
  switch (pde.num_dims())
  {
  case 2:
    createMomentReducedMatrix_nd<1>(pde, hash_table);
    break;
  case 3:
    createMomentReducedMatrix_nd<2>(pde, hash_table);
    break;
  case 4:
    createMomentReducedMatrix_nd<3>(pde, hash_table);
    break;
  default:
    throw std::runtime_error(
        "unsupported number of dimensions with createMomentReducedMatrix");
  }
}

template<typename P>
template<int nvdim>
void moment<P>::createMomentReducedMatrix_nd(PDE<P> const &pde,
                                             elements::table const &hash_table)
{
  int const num_ele = hash_table.size();

  int const moment_idx = 0;
  int const x_dim      = 0; // hardcoded for now, needs to change
  int const v_dim_1    = 1;

  expect(static_cast<int>(this->fList.size()) > moment_idx);
  expect(this->fList[moment_idx].size() >= nvdim);
  auto g_vec_1 = this->fList[moment_idx][v_dim_1];

  // Define additional g_vecs for higher number of v dimensions
  fk::vector<P> g_vec_2, g_vec_3;
  if (nvdim >= 2)
  {
    g_vec_2.resize(this->fList[moment_idx][2].size()) =
        this->fList[moment_idx][2];
    if (nvdim >= 3)
    {
      g_vec_3.resize(this->fList[moment_idx][3].size()) =
          this->fList[moment_idx][3];
    }
  }

  expect(pde.get_dimensions().size() == nvdim + 1);
  int const n = fm::ipow(pde.get_dimensions()[v_dim_1].get_degree() + 1, nvdim + 1) *
                num_ele;
  auto const &dim = pde.get_dimensions()[x_dim];
  int const rows  = fm::ipow2(dim.get_level()) * (dim.get_degree() + 1);

  std::multimap<int, dense_item<P>> moment_mat;

  int const pdof = pde.get_dimensions()[v_dim_1].get_degree() + 1;

  // TODO: this should be refactored into a sparse matrix
  for (int i = 0; i < num_ele; i++)
  {
    // l_dof_x and l_dof_v
    fk::vector<int> const coords       = hash_table.get_coords(i);
    fk::vector<int> const elem_indices = linearize(coords);

    for (int j = 0; j < pdof; j++)
    {
      int const ind_i = elem_indices(x_dim) * pdof + j; // row_idx
      for (int vdeg1 = 0; vdeg1 < pdof; vdeg1++)
      {
        if (nvdim == 1)
        {
          // "2D" case (v_dim = 1)
          int const ind_j = i * fm::ipow(pdof, 2) + j * pdof;
          moment_mat.insert(
              {ind_i, dense_item<P>{ind_i, ind_j + vdeg1,
                                    g_vec_1(elem_indices(1) * pdof + vdeg1)}});
        }
        else
        {
          for (int vdeg2 = 0; vdeg2 < pdof; vdeg2++)
          {
            if (nvdim == 2)
            {
              // "3D" case (v_dim = 2)
              int const ind_j = i * fm::ipow(pdof, 3) +
                                j * fm::ipow(pdof, 2) +
                                pdof * vdeg1 + vdeg2;
              moment_mat.insert(
                  {ind_i,
                   dense_item<P>{ind_i, ind_j,
                                 g_vec_1(elem_indices(1) * pdof + vdeg1) *
                                     g_vec_2(elem_indices(2) * pdof + vdeg2)}});
            }
            else if (nvdim == 3)
            {
              // "4D" case (v_dim = 3)
              for (int vdeg3 = 0; vdeg3 < pdof; vdeg3++)
              {
                int const ind_j = i * fm::ipow(pdof, 4) +
                                  j * fm::ipow(pdof, 3) +
                                  fm::ipow(pdof, 2) * vdeg1 +
                                  vdeg2 * pdof + vdeg3;
                moment_mat.insert(
                    {ind_i, dense_item<P>{
                                ind_i, ind_j,
                                g_vec_1(elem_indices(1) * pdof + vdeg1) *
                                    g_vec_2(elem_indices(2) * pdof + vdeg2) *
                                    g_vec_3(elem_indices(3) * pdof + vdeg3)}});
              }
            }
          }
        }
      }
    }
  }

  // TODO: sparse construction is host-only
  fk::sparse<P, resource::host> host_sparse =
      fk::sparse<P, resource::host>(moment_mat, n, rows);
  if constexpr (sparse_resrc == resource::device)
  {
    // create a sparse version of this matrix and put it on the GPU
    this->sparse_mat = host_sparse.clone_onto_device();
  }
  else
  {
    this->sparse_mat = host_sparse;
  }
}

template<typename P>
fk::vector<P> &moment<P>::create_realspace_moment(
    PDE<P> const &pde_1d, fk::vector<P> &wave, elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace)
{
  // the realspace size uses the number of quadrature points (10) rather than
  // degree
  int const realspace_size =
      ASGARD_NUM_QUADRATURE *
      fm::ipow2(pde_1d.get_dimensions()[0].get_level());
  this->realspace = fk::vector<P>(realspace_size);
  wavelet_to_realspace<P>(pde_1d, wave, table, transformer, workspace,
                          this->realspace, quadrature_mode::use_fixed);
  return this->realspace;
}

#ifdef ASGARD_USE_CUDA
template<typename P>
fk::vector<P> &moment<P>::create_realspace_moment(
    PDE<P> const &pde_1d,
    fk::vector<P, mem_type::owner, resource::device> &wave,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace)
{
  fk::vector<P> wave_host = wave.clone_onto_host();
  // the realspace size uses the number of quadrature points (10) rather than
  // degree
  int const realspace_size =
      ASGARD_NUM_QUADRATURE *
      fm::ipow2(pde_1d.get_dimensions()[0].get_level());
  this->realspace = fk::vector<P>(realspace_size);
  wavelet_to_realspace<P>(pde_1d, wave_host, table, transformer, workspace,
                          this->realspace, quadrature_mode::use_fixed);
  return this->realspace;
}
#endif

#ifdef ASGARD_ENABLE_DOUBLE
template class moments1d<double>;
template class moment<double>;
#endif
#ifdef ASGARD_ENABLE_FLOAT
template class moments1d<float>;
template class moment<float>;
#endif

} // namespace asgard
