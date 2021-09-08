#include "cblacs_grid.hpp"
#include "distribution.hpp"
#include "tools.hpp"

#include <cmath>

extern "C"
{
  int numroc_(int *, int *, int *, int *, int *);
  void Cblacs_get(int, int, int *);
  void Cblacs_gridinit(int *, const char *, int, int);
  void Cblacs_gridinfo(int, int *, int *, int *, int *);
  void Cblacs_gridexit(int);
  void Cblacs_pinfo(int *, int *);
  void Cblacs_exit(int);
}

cblacs_grid::cblacs_grid()
{
  int i_negone{-1}, i_zero{0};
  int myid, numproc;
  Cblacs_pinfo(&myid, &numproc);
  npcol_ = get_num_subgrid_cols(numproc);
  nprow_ = numproc / npcol_;
  expect((nprow_ >= 1) && (npcol_ >= 1) && (nprow_ * npcol_ == numproc));
  Cblacs_get(i_negone, i_zero, &ictxt_);
  Cblacs_gridinit(&ictxt_, "R", nprow_, npcol_);
  Cblacs_gridinfo(ictxt_, &nprow_, &npcol_, &myrow_, &mycol_);
}

int cblacs_grid::local_rows(int m, int mb)
{
  int i_zero{0};
  return numroc_(&m, &mb, &myrow_, &i_zero, &nprow_);
}

int cblacs_grid::local_cols(int n, int nb)
{
  int i_zero{0};
  return numroc_(&n, &nb, &mycol_, &i_zero, &npcol_);
}

cblacs_grid::~cblacs_grid() { Cblacs_gridexit(ictxt_); }
