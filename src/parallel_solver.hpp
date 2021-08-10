#pragma once

enum DESC_VARS : int
{
  DTYPE_ = 0,
  CTXT_  = 1,
  M_     = 2,
  N_     = 3,
  MB_    = 4,
  NB_    = 5,
  RSRC_  = 6,
  CSRC_  = 7,
  LLD_   = 8,
  DLEN_  = 9
};

namespace parallel_solver
{
template<typename P>
void gather_matrix(P *A, int *descA, P *A_distr, int *descA_distr);
template<typename P>
void scatter_matrix(P *A, int *descA, P *A_distr, int *descA_distr);
} // namespace parallel_solver
