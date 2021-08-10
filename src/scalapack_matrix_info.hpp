#pragma once
#include <array>
#include <memory>

class cblacs_grid;

namespace fk
{
class scalapack_matrix_info
{
public:
  scalapack_matrix_info(int rows, int cols);
  scalapack_matrix_info(int rows, int cols, int mb, int nb,
                        std::shared_ptr<cblacs_grid> grid);
  void resize(int rows, int cols);
  int *get_desc() { return desc_.data(); }
  const int *get_desc() const { return desc_.data(); }
  int mb() const { return mb_; }
  int nb() const { return nb_; }
  int nrows() const { return rows_; }
  int ncols() const { return cols_; }
  int local_rows() const { return local_rows_; }
  int local_cols() const { return local_cols_; }
  std::shared_ptr<cblacs_grid> get_grid() const { return grid_; }

private:
  int rows_, cols_;
  int local_rows_, local_cols_;
  int mb_, nb_;
  std::array<int, 9> desc_;
  std::shared_ptr<cblacs_grid> grid_;
};

} // namespace fk
