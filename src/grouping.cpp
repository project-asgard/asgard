#include "grouping.hpp"

template<typename P>
rank_workspace<P>::rank_workspace(PDE<P> const &pde,
                                  std::vector<element_group> const &groups)
{
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));

  int const max_elems =
      (*std::max_element(groups.begin(), groups.end(),
                         [&](const element_group &a, const element_group &b) {
                           return a.size() < b.size();
                         }))
          .size();

  int const max_conn = [&groups]() {
    int current_max = 0;
    for (element_group const &group : groups)
    {
      for (const auto &[row, cols] : group)
      {
        current_max = std::max(current_max, cols.second - cols.first + 1);
      }
    }
    return current_max;
  }();

  auto const get_elems_in_group = [](element_group const &g) {
    int num_elems = 0;
    for (const auto &[row, cols] : g)
    {
      num_elems += cols.second - cols.first + 1;
    }
    return num_elems;
  };
  int const max_total = get_elems_in_group(
      *std::max_element(groups.begin(), groups.end(),
                        [&](const element_group &a, const element_group &b) {
                          return get_elems_in_group(a) < get_elems_in_group(b);
                        }));

  batch_input.resize(elem_size * max_conn);
  batch_output.resize(elem_size * max_elems);
  reduction_space.resize(elem_size * max_total * pde.num_terms);

  // intermediate workspaces for kron product.
  int const num_workspaces = std::min(pde.num_dims - 1, 2);
  batch_intermediate.resize(reduction_space.size() * num_workspaces);
  unit_vector_.resize(pde.num_terms * max_conn);
  std::fill(unit_vector_.begin(), unit_vector_.end(), 1.0);
}

template<typename P>
fk::vector<P> const &rank_workspace<P>::get_unit_vector() const
{
  return unit_vector_;
}

// calculate how much workspacespace we need on device for single connected
// element
//
// *does not include operator matrices - working for now on assumption they'll
// all be resident*
template<typename P>
static double get_element_size_MB(PDE<P> const &pde)
{
  auto const get_MB = [](auto const num_elems) -> double {
    assert(num_elems > 0);
    double const bytes     = num_elems * sizeof(P);
    double const megabytes = bytes * 1e-6;
    return megabytes;
  };

  // FIXME assume uniform degree
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  // number of intermediate workspaces for kron product.
  // FIXME this only applies to explicit
  int const num_workspaces = std::min(pde.num_dims - 1, 2);

  // calc size of reduction space for a single work item
  double const elem_reduction_space_MB = get_MB(pde.num_terms * elem_size);
  // calc size of intermediate space for a single work item
  double const elem_intermediate_space_MB =
      get_MB(static_cast<double>(num_workspaces) * pde.num_terms * elem_size);

  // calc in and out vector sizes for each elem
  // since we will scheme to have most elems require overlapping pieces of
  // x and y, we will never need 2 addtl xy space per elem
  double const elem_xy_space_MB = get_MB(elem_size * 1.2);
  return (elem_reduction_space_MB + elem_intermediate_space_MB +
          elem_xy_space_MB);
}

// determine how many groups will be required to solve the problem
// a task is a subset of all elements whose total workspace requirement
// is less than the limit passed in rank_size_MB
template<typename P>
int get_num_groups(element_table const &table, PDE<P> const &pde,
                   int const num_ranks, int const rank_size_MB)
{
  assert(num_ranks > 0);
  assert(rank_size_MB > 0);
  // determine total problem size
  double const num_elems = static_cast<double>(table.size()) * table.size();
  double const space_per_elem = get_element_size_MB(pde);
  // make sure rank size is something reasonable
  // a single element is the finest we can split the problem
  // if that requires a lot of space relative to rank size,
  // roundoff of elements over groups will cause us to exceed the limit
  //
  // also not feasible to solve problem with such a workspace limit

  assert(space_per_elem < (0.5 * rank_size_MB));
  double const problem_size_MB = space_per_elem * num_elems;

  // determine number of groups
  double const problem_size_per_rank = problem_size_MB / rank_size_MB;
  int const num_groups               = [problem_size_per_rank, num_ranks] {
    int const groups_per_rank =
        static_cast<int>(problem_size_per_rank / num_ranks + 1);
    return groups_per_rank * num_ranks;
  }();
  return num_groups;
}

// divide the problem given the previously computed number of groups
// this function divides via a greedy, row-major split.
std::vector<element_group>
assign_elements(element_table const &table, int const num_groups)
{
  assert(num_groups > 0);

  int64_t const num_elems = static_cast<int64_t>(table.size()) * table.size();

  int64_t const elems_left_over = num_elems % num_groups;
  int64_t const elems_per_task =
      num_elems / num_groups + elems_left_over / num_groups;
  int64_t const still_left_over = elems_left_over % num_groups;

  std::vector<element_group> grouping;
  int64_t assigned = 0;

  for (int i = 0; i < num_groups; ++i)
  {
    std::map<int, std::vector<int>> group_map;

    auto const insert = [&group_map](int const key, int col) {
      group_map.try_emplace(key, std::vector<int>());
      group_map[key].push_back(col);
    };
    int64_t const elems_this_task =
        i < still_left_over ? elems_per_task + 1 : elems_per_task;
    int64_t const task_end = assigned + elems_this_task - 1;

    int64_t const group_start_row = assigned / table.size();
    int64_t const group_start_col = assigned % table.size();
    int64_t const group_end_row   = task_end / table.size();
    int64_t const group_end_col   = task_end % table.size();

    assigned += elems_this_task;

    if (group_end_row > group_start_row)
    {
      for (int i = group_start_row + 1; i < group_end_row; ++i)
      {
        for (int j = 0; j < table.size(); ++j)
        {
          insert(i, j);
        }
      }
      for (int j = group_start_col; j < table.size(); ++j)
      {
        insert(group_start_row, j);
      }
      for (int j = 0; j <= group_end_col; ++j)
      {
        insert(group_end_row, j);
      }
    }
    else
    {
      for (int j = group_start_col; j <= group_end_col; ++j)
      {
        insert(group_start_row, j);
      }
    }

    element_group group;
    for (const auto &[row, cols] : group_map)
    {
      group[row] = std::make_pair(cols[0], cols.back());
    }
    grouping.push_back(group);
  }
  return grouping;
}

template class rank_workspace<float>;
template class rank_workspace<double>;

template int get_num_groups(element_table const &table, PDE<float> const &pde,
                            int const num_ranks, int const rank_size_MB);
template int get_num_groups(element_table const &table, PDE<double> const &pde,
                            int const num_ranks, int const rank_size_MB);
