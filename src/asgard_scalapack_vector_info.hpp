#pragma once
#include "asgard_scalapack_matrix_info.hpp"

namespace asgard::fk
{
class scalapack_vector_info
{
public:
  scalapack_vector_info(int size) : info_(size, 1) {}
  scalapack_vector_info(int size, int mb, std::shared_ptr<cblacs_grid> grid)
      : info_(size, 1, mb, 1, std::move(grid))
  {}
  void resize(int new_size) { info_.resize(new_size, 1); }
  int *get_desc() { return info_.get_desc(); }
  const int *get_desc() const { return info_.get_desc(); }
  int mb() const { return info_.mb(); }
  int size() const { return info_.nrows(); }
  int local_size() const { return info_.local_rows(); }
  std::shared_ptr<cblacs_grid> get_grid() const { return info_.get_grid(); }

private:
  scalapack_matrix_info info_;
};
} // namespace asgard::fk
