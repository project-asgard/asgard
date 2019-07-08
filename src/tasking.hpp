#pragma once
#include "element_table.hpp"
#include "pde.hpp"
#include "tensors.hpp"

class task
{
public:
  task(int const elem_start, int const elem_end, int const conn_start,
       int const conn_end)
      : elem_start(elem_start), elem_end(elem_end), conn_start(conn_start),
        conn_end(conn_end)

  {}
  task(task const &other)
      : elem_start(other.elem_start), elem_end(other.elem_end),
        conn_start(other.conn_start), conn_end(other.conn_end)
  {}
  task(task const &&other)
      : elem_start(other.elem_start), elem_end(other.elem_end),
        conn_start(other.conn_start), conn_end(other.conn_end)
  {}

  int const elem_start;
  int const elem_end;
  int const conn_start;
  int const conn_end;
};

template<typename P>
class task_workspace
{
public:
  task_workspace(PDE<P> const &pde, element_table const &table,
                 std::vector<task> const &tasks);

  fk::vector<P> const &get_unit_vector() const;
  // input, output, workspace for batched gemm/reduction
  // (unit vector below also falls under this category)
  fk::vector<P> batch_input;
  fk::vector<P> reduction_space;
  fk::vector<P> batch_intermediate;
  fk::vector<P> batch_output;
  double size_MB() const
  {
    int64_t num_elems = batch_input.size() + reduction_space.size() +
                        batch_intermediate.size() + batch_output.size() +
                        unit_vector_.size();
    double const bytes     = static_cast<double>(num_elems) * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

private:
  fk::vector<P> unit_vector_;
};

template<typename P>
int get_num_tasks(element_table const &table, PDE<P> const &pde,
                  int const num_ranks, int const rank_size_MB);

// old, phase out
std::vector<task>
assign_elements_to_tasks(element_table const &table, int const num_tasks);

// new, phase in
using task_map = std::map<int, std::vector<int>>;
task_map assign_elements(element_table const &table, int const num_tasks);

extern template int get_num_tasks(element_table const &table,
                                  PDE<float> const &pde, int const num_ranks,
                                  int const rank_size_MB);
extern template int get_num_tasks(element_table const &table,
                                  PDE<double> const &pde, int const num_ranks,
                                  int const rank_size_MB);
