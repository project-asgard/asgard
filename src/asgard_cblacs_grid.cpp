#include "asgard_tools.hpp"

#include "asgard_cblacs_grid.hpp"
#include "asgard_distribution.hpp"

extern "C"
{
  int numroc_(int *, int *, int *, int *, int *);
  void Cblacs_get(int, int, int *);
  int Csys2blacs_handle(MPI_Comm);
  void Cblacs_gridinit(int *, const char *, int, int);
  void Cblacs_gridinfo(int, int *, int *, int *, int *);
  void Cfree_blacs_system_handle(int);
  void Cblacs_gridexit(int);
  void Cblacs_pinfo(int *, int *);
  void Cblacs_exit(int);
}
namespace asgard
{
cblacs_grid::cblacs_grid(MPI_Comm localCommunicator)
{
  int i_negone{-1}, i_zero{0};
  int numproc = get_num_ranks();
  npcol_      = get_num_subgrid_cols(numproc);
  nprow_      = numproc / npcol_;
  expect((nprow_ >= 1) && (npcol_ >= 1) && (nprow_ * npcol_ == numproc));
  Cblacs_get(i_negone, i_zero, &ictxt_);
  bhandle_ = Csys2blacs_handle(localCommunicator);
  ictxt_   = bhandle_;
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

cblacs_grid::~cblacs_grid()
{
  Cfree_blacs_system_handle(bhandle_);
  Cblacs_gridexit(ictxt_);
}
} // namespace asgard
