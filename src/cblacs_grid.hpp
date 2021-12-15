#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "mpi.h"
#pragma GCC diagnostic pop

class cblacs_grid
{
public:
  cblacs_grid(MPI_Comm localCommunicator);
  int get_context() const { return ictxt_; }
  int get_myrow() const { return myrow_; }
  int get_mycol() const { return mycol_; }
  int local_rows(int m, int mb);
  int local_cols(int n, int nb);
  ~cblacs_grid();

private:
  int bhandle_, ictxt_, nprow_{1}, npcol_, myrow_, mycol_;
};
