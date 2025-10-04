#include "asgard_grid_1d.hpp"

namespace asgard
{
#ifdef ASGARD_USE_GPU
void gpu_connect_1d::add_level(connect_1d const &conn, conn_fill fill)
{
  int nnz = conn.num_connections();
  if (fill != conn_fill::both) {
    // must count the true number of non-zeros
    nnz = 0;
    if (fill == conn_fill::upper) {
      for (int r = 0; r < conn.num_rows(); r++)
        nnz += conn.row_end(r) - conn.row_diag(r);
    } else {
      for (int r = 0; r < conn.num_rows(); r++)
        nnz += conn.row_diag(r) - conn.row_begin(r);
    }
  }

  std::vector<int> rc;
  rc.reserve(3 * nnz);

  for (int r = 0; r < conn.num_rows(); r++) {
    int const rbegin = (fill == conn_fill::upper) ? conn.row_diag(r) : conn.row_begin(r);
    int const rend   = (fill == conn_fill::lower) ? conn.row_diag(r) : conn.row_end(r);
    for (int j = rbegin; j < rend; j++) {
      rc.push_back(r);
      rc.push_back(conn[j]);
      rc.push_back(j);
    }
  }

  lrowcol.emplace_back(rc);
}

void gpu_connect_1d::done_adding()
{
  std::vector<int> nz;
  nz.reserve(lrowcol.size());

  std::vector<int *> rc;
  rc.reserve(lrowcol.size());

  for (auto &v : lrowcol) {
    nz.push_back(static_cast<int>(v.size()) / 3);
    rc.push_back(v.data());
  }

  nnz_    = nz;
  rowcol_ = rc;
}

void connection_patterns::load_to_gpu()
{
  int const num_gpus  = compute->num_gpus();
  int const max_level = conns[0].max_loaded_level();

  int const lend = max_level + 1;
  lconns[0].resize(max_level + 1);
  lconns[1].resize(max_level + 1);
  #pragma omp parallel for
  for (int l = 0; l < 2 * lend; l++) {
    lconns[l % 2][l / 2]
      = connect_1d(l/ 2, (l % 2 == 0) ? connect_1d::hierarchy::volume : connect_1d::hierarchy::full);
  }

  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    gpu_conns[g] = gpu_connect(max_level);
  }
}
#endif

} // namespace asgard
