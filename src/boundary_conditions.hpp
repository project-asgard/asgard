#pragma once
#include "element_table.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "pde/pde_base.hpp"
#include "transformations.hpp"

template<typename P>
class bc_timestepper
{
public:
  bc_timestepper(PDE<P> const &pde, element_table const &table,
                 int const start_element, int const stop_element,
                 P const t_init = 0);

  fk::vector<P> advance(P const time) const;

  void print_left() const;

  void print_right() const;

  fk::vector<P> compute_left_boundary_condition(g_func_type const g_func,
                                                P const time, int level,
                                                int degree, P const domain_min,
                                                P const domain_max,
                                                vector_func<P> const bc_func);

  fk::vector<P> compute_right_boundary_condition(g_func_type const g_func,
                                                 P const time, int level,
                                                 int degree, P const domain_min,
                                                 P const domain_max,
                                                 vector_func<P> const bc_func);

private:
  std::vector<std::vector<std::vector<fk::vector<P>>>> left;

  std::vector<std::vector<std::vector<fk::vector<P>>>> right;

  P const t_init;

  int const bc_size;

  PDE<P> const &pde;

  std::vector<fk::vector<P>>
  generate_partial_bcs(std::vector<dimension<P>> const &dimensions,
                       int const d_index,
                       std::vector<vector_func<P>> const &bc_funcs,
                       P const time,
                       std::vector<partial_term<P>> const &partial_terms,
                       int const p_index, fk::vector<P> &&trace_bc);
};
