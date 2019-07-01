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

private:
  fk::vector<P> unit_vector_;
};

template<typename P>
int get_num_tasks(element_table const &table, PDE<P> const &pde,
                  int const num_ranks, int const rank_size_MB);

std::vector<task>
assign_elements_to_tasks(element_table const &table, int const num_tasks);

extern template int get_num_tasks(element_table const &table,
                                  PDE<float> const &pde, int const num_ranks,
                                  int const rank_size_MB);
extern template int get_num_tasks(element_table const &table,
                                  PDE<double> const &pde, int const num_ranks,
                                  int const rank_size_MB);

/*
class task_list
{

public:

  template<typename P>
  task_list(element_table const &table, PDE<P> const &pde, int const num_ranks,
         int const rank_size_MB);

private:

  std::vector<std::vector<task>> tasks_by_rank;
  std::vector<task> get_tasks(int const rank_num);

};*/
