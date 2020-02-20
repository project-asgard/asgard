#include "solver.hpp"

template<typename P>
P gmres(PDE<P> const &pde, element_table const &elem_table,
        distribution_plan const &plan, std::vector<element_chunk> const &chunks,
        host_workspace<P> &host_space, rank_workspace<P> &rank_space,
        std::vector<batch_operands_set<P>> &batches, P const dt,
        P const threshold, int const restart)
{
  assert(restart > 0);
  auto const degree      = pde.get_dimensions()[0].get_degree();
  auto const elem_size   = static_cast<int>(std::pow(degree, pde.num_dims));
  auto const system_size = elem_size * static_cast<int64_t>(table.size());
  assert(restart < system_size);

  assert(threshold > 0.0);
  assert(theshold < 1.0);

  int const my_rank = get_rank();

  // allocate space
  fk::matrix<P> basis(host_space.x.size(), restart);
  fk::matrix<P> subspace_projection(restart + 1, restart);
  fk::vector<P> cosines(restart);
  fk::vector<P> sines(restart);

  // initial residual -- may skip this eventually
  fm::copy(host_space.x, host_space.x_orig);
  apply_A(pde, elem_table, plan.at(my_rank), chunks, host_space, rank_space,
          batches);
  reduce_results(host_space.fx, host_space.reduced_fx, plan, my_rank);
  exchange_results(host_space.reduced_fx, host_space.x, elem_size, plan,
                   my_rank);
}

template float
gmres(PDE<float> const &pde, element_table const &elem_table,
      distribution_plan const &plan, std::vector<element_chunk> const &chunks,
      host_workspace<float> &host_space, rank_workspace<float> &rank_space,
      std::vector<batch_operands_set<float>> &batches, float const dt);

template double
gmres(PDE<double> const &pde, element_table const &elem_table,
      std::vector<element_chunk> const &chunks, distribution_plan const &plan,
      host_workspace<double> &host_space, rank_workspace<double> &rank_space,
      std::vector<batch_operands_set<double>> &batches, double const dt);
