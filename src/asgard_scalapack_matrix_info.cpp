#include "asgard_scalapack_matrix_info.hpp"
#include "asgard_cblacs_grid.hpp"
#include <stdexcept>

extern "C"
{
  void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
                 int *icsrc, int *ictxt, int *lld, int *info);
}

namespace asgard::fk
{
scalapack_matrix_info::scalapack_matrix_info(int rows, int cols)
    : rows_{rows}, cols_{cols}, local_rows_{rows}, local_cols_{cols}, mb_{rows},
      nb_{cols}, desc_{{1, 0, rows_, cols_, rows_, cols_, 0, 0, rows_}}
{}

scalapack_matrix_info::scalapack_matrix_info(int rows, int cols, int mb, int nb,
                                             std::shared_ptr<cblacs_grid> grid)
    : rows_{rows}, cols_{cols}, mb_{mb}, nb_{nb}, grid_{std::move(grid)}
{
  if (grid_)
  {
    int i_zero{0}, info;
    int ictxt = grid_->get_context();
    int lld   = std::max(1, grid_->local_rows(rows_, mb_));
    descinit_(desc_.data(), &rows_, &cols_, &mb_, &nb_, &i_zero, &i_zero,
              &ictxt, &lld, &info);
    local_rows_ = grid_->local_rows(rows_, mb_);
    local_cols_ = grid_->local_cols(cols_, nb_);
  }
  else
  {
    throw std::invalid_argument("cblas_grid pointer is null!");
  }
}

void scalapack_matrix_info::resize(int rows, int cols)
{
  rows_ = rows;
  cols_ = cols;
  if (grid_)
  {
    int i_zero{0}, info;
    int ictxt = grid_->get_context();
    int lld   = std::max(1, grid_->local_rows(rows_, mb_));
    descinit_(desc_.data(), &rows_, &cols_, &mb_, &nb_, &i_zero, &i_zero,
              &ictxt, &lld, &info);
    local_rows_ = grid_->local_rows(rows_, mb_);
    local_cols_ = grid_->local_cols(cols_, nb_);
  }
  else
  {
    desc_       = {{1, 0, rows_, cols_, rows_, cols_, 0, 0, rows_}};
    local_rows_ = rows_;
    local_cols_ = cols_;
  }
}
} // namespace asgard::fk
