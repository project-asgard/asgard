#include "moment.hpp"
#include "basis.hpp"
#include "elements.hpp"
#include "sparse.hpp"
#include "tools.hpp"
#include "transformations.hpp"

namespace asgard
{
template<typename P>
moment<P>::moment(std::vector<md_func_type<P>> md_funcs_)
    : md_funcs(std::move(md_funcs_))
{}

// Creates the coefficients of the moment vector on each domain.
// No mass matrix inversion is done.
template<typename P>
void moment<P>::createFlist(PDE<P> const &pde, options const &opts)
{
  std::size_t num_md_funcs = this->md_funcs.size();

  auto const &dims     = pde.get_dimensions();
  std::size_t num_dims = dims.size();

  this->fList.clear();
  this->fList.resize(num_md_funcs);

  basis::wavelet_transform<P, resource::host> const transformer(opts, pde);

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
void moment<P>::createMomentVector(PDE<P> const &pde, options const &opts,
                                   elements::table const &hash_table)
{
  // check that fList has been constructed
  expect(this->fList.size() > 0);

  if (this->vector.empty() || opts.do_adapt_levels)
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

// helpers for converting linear coordinates into operator matrix indices
inline fk::vector<int> linearize(fk::vector<int> const &coords)
{
  fk::vector<int> linear(coords.size() / 2);
  for (int i = 0; i < linear.size(); ++i)
  {
    linear(i) = elements::get_1d_index(coords(i), coords(i + linear.size()));
  }
  return linear;
}

template<typename P>
inline fk::vector<int>
linear_coords_to_indices(PDE<P> const &pde, int const degree,
                         fk::vector<int> const &coords)
{
  fk::vector<int> indices(coords.size());
  for (int d = 0; d < pde.num_dims; ++d)
  {
    indices(d) = coords(d) * degree;
  }
  return indices;
}

template<typename P>
void moment<P>::createMomentReducedMatrix(PDE<P> const &pde,
                                          elements::table const &hash_table)
{
  switch (pde.num_dims)
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
    g_vec_2 = this->fList[moment_idx][2];
    if (nvdim >= 3)
    {
      g_vec_3 = this->fList[moment_idx][3];
    }
  }

  expect(pde.get_dimensions().size() == nvdim + 1);
  int const n = static_cast<int>(std::pow(
                    pde.get_dimensions()[v_dim_1].get_degree(), nvdim + 1)) *
                num_ele;
  int const rows =
      static_cast<int>(std::pow(2, pde.get_dimensions()[x_dim].get_level())) *
      pde.get_dimensions()[x_dim].get_degree();

  if (this->moment_matrix.size() != rows * n)
  {
    this->moment_matrix.clear_and_resize(rows, n);
  }

  int const deg = pde.get_dimensions()[v_dim_1].get_degree();

  // TODO: this should be refactored into a sparse matrix
  for (int i = 0; i < num_ele; i++)
  {
    // l_dof_x and l_dof_v
    fk::vector<int> const coords       = hash_table.get_coords(i);
    fk::vector<int> const elem_indices = linearize(coords);

    for (int j = 0; j < deg; j++)
    {
      int const ind_i = elem_indices(x_dim) * deg + j; // row_idx
      for (int vdeg1 = 0; vdeg1 < deg; vdeg1++)
      {
        if (nvdim == 1)
        {
          // "2D" case (v_dim = 1)
          int const ind_j = i * static_cast<int>(std::pow(deg, 2)) + j * deg;
          moment_matrix(ind_i, ind_j + vdeg1) =
              g_vec_1(elem_indices(1) * deg + vdeg1);
        }
        else
        {
          for (int vdeg2 = 0; vdeg2 < deg; vdeg2++)
          {
            if (nvdim == 2)
            {
              // "3D" case (v_dim = 2)
              int const ind_j = i * static_cast<int>(std::pow(deg, 3)) +
                                j * static_cast<int>(std::pow(deg, 2)) +
                                deg * vdeg1 + vdeg2;
              moment_matrix(ind_i, ind_j) =
                  g_vec_1(elem_indices(1) * deg + vdeg1) *
                  g_vec_2(elem_indices(2) * deg + vdeg2);
            }
            else if (nvdim == 3)
            {
              // "4D" case (v_dim = 3)
              for (int vdeg3 = 0; vdeg3 < deg; vdeg3++)
              {
                int const ind_j = i * static_cast<int>(std::pow(deg, 4)) +
                                  j * static_cast<int>(std::pow(deg, 3)) +
                                  static_cast<int>(std::pow(deg, 2)) * vdeg1 +
                                  vdeg2 * deg + vdeg3;
                moment_matrix(ind_i, ind_j) =
                    g_vec_1(elem_indices(1) * deg + vdeg1) *
                    g_vec_2(elem_indices(2) * deg + vdeg2) *
                    g_vec_3(elem_indices(3) * deg + vdeg3);
              }
            }
          }
        }
      }
    }
  }

  // create a sparse version of this matrix and put it on the GPU
  this->sparse_mat =
      fk::sparse<P, mem_type::owner, resource::host>(this->moment_matrix)
          .clone_onto_device();
  std::cout << this->moment_matrix.size() << "\n";
  std::cout << this->sparse_mat.sp_size() << "\n";
}

template<typename P>
fk::vector<P> &moment<P>::create_realspace_moment(
    PDE<P> const &pde_1d, fk::vector<P> &wave, elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace)
{
  this->realspace.resize(wave.size());
  wavelet_to_realspace<P>(pde_1d, wave, table, transformer, workspace,
                          this->realspace);
  return this->realspace;
}

template<typename P>
fk::vector<P> &moment<P>::create_realspace_moment(
    PDE<P> const &pde_1d,
    fk::vector<P, mem_type::owner, resource::device> &wave,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace)
{
  fk::vector<P> wave_host = wave.clone_onto_host();
  this->realspace.resize(wave_host.size());
  wavelet_to_realspace<P>(pde_1d, wave_host, table, transformer, workspace,
                          this->realspace);
  return this->realspace;
}

#ifdef ASGARD_ENABLE_DOUBLE
template class moment<double>;
#endif
#ifdef ASGARD_ENABLE_FLOAT
template class moment<float>;
#endif

} // namespace asgard
