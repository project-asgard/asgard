#include "asgard_grid_1d.hpp"

namespace asgard
{
#ifdef ASGARD_GPU_NON_GREEDY
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

  lrowcol.emplace_back(std::move(rc));
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

gpu_connect::gpu_connect(int max_level)
{
  for (int l = 0; l <= max_level; l++)
  {
    {
      connect_1d const conn(l, connect_1d::hierarchy::volume);

      for (int p = 0; p < 3; p++)
        patts[p].add_level(conn, static_cast<conn_fill>(p));
    }{
      connect_1d const conn(l, connect_1d::hierarchy::full);
      full().add_level(conn, conn_fill::both);
    }
  }

  for (auto &p : patts)
    p.done_adding();
}

gpu_connect::gpu_connect(std::vector<connect_1d> &levels, connect_1d const &last,
                         connect_1d::hierarchy h)
{
  expect(h == connect_1d::hierarchy::volume);
  for (auto const &lvl : levels) {
    for (int p = 0; p < 3; p++)
      patts[p].add_level(lvl, static_cast<conn_fill>(p));
  }
  for (int p = 0; p < 3; p++)
    patts[p].add_level(last, static_cast<conn_fill>(p));

  for (int p = 0; p < 3; p++)
    patts[p].done_adding();
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

void connection_patterns::load_reduced_fill()
{
  connect_1d const &conn = conns[static_cast<int>(connect_1d::hierarchy::volume)];
  int const max_level = conn.max_loaded_level();
  lconns[0].reserve(max_level);

  int rows = 1; // number of cells on this level
  for (int l = 0; l < max_level; l++)
  {
    int outj = 0;
    for (int r = 0; r < rows; r++)
    {
      outj = conn.row_diag(r) - conn.row_begin(r) + 1;
      for (int j = conn.row_diag(r) + 1; j < conn.row_end(r); j++) {
        if (conn[j] >= rows)
          break;
        outj++;
      }
    }

    std::vector<int> pntr;  pntr.reserve(rows + 1);
    std::vector<int> indx;  indx.reserve(outj);
    std::vector<int> diag;  diag.reserve(outj);

    pntr.push_back(0);
    for (int r = 0; r < rows; r++)
    {
      indx.insert(indx.end(), conn.get_indx().begin() + conn.row_begin(r),
                              conn.get_indx().begin() + conn.row_diag(r));

      diag.push_back(static_cast<int>(indx.size()));
      indx.push_back(r);

      for (int j = conn.row_diag(r) + 1; j < conn.row_end(r); j++) {
        if (conn[j] >= rows)
          break;
        indx.push_back(conn[j]);
      }

      pntr.push_back(static_cast<int>(indx.size()));
    }

    lconns[0].emplace_back(l, rows, std::move(pntr), std::move(indx), std::move(diag));

    rows *= 2;
  }

  int const num_gpus  = compute->num_gpus();
  #pragma omp parallel for schedule(static, 1)
  for (int g = 0; g < num_gpus; g++) {
    compute->set_device(gpu::device{g});
    gpu_conns[g] = gpu_connect(lconns[0], conn, connect_1d::hierarchy::volume);
  }
}
#endif

} // namespace asgard
