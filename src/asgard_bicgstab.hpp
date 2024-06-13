#pragma once
#include "asgard_kronmult_matrix.hpp"
#include "asgard_matrix.hpp"
#include "asgard_vector.hpp"
#include "batch.hpp"
#include "pde.hpp"

namespace asgard::solver
{
// simple, node-local test version of gmres
template<typename P>
bicgstab_info<P>
bicgstab(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
         fk::matrix<P> const &M, int const restart, int const max_iter,
         P const tolerance);

#ifdef KRON_MODE_GLOBAL
// solves ( I - dt * mat ) * x = b
template<typename P, resource resrc>
gmres_info<P>
bicgstab_euler(const P dt, matrix_entry mentry,
               global_kron_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const restart, int const max_iter, P const tolerance);
#else
// solves ( I - dt * mat ) * x = b
template<typename P, resource resrc>
bicgstab_info<P>
bicgstab_euler(const P dt, kronmult_matrix<P> const &mat,
               fk::vector<P, mem_type::owner, resrc> &x,
               fk::vector<P, mem_type::owner, resrc> const &b,
               int const restart, int const max_iter, P const tolerance);
#endif

} // namespace asgard::solver