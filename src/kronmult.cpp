#include "kronmult.hpp"
#include "device/kronmult_cuda.hpp"
#include "lib_dispatch.hpp"
#include "tools.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <limits.h>
#include <mutex>
#include <vector>

namespace kronmult
{

 /* Data conversion for adaptivity probing time step Mixed Precision Optim */
  void allocate_sp_space(
          int64_t work_size, int64_t output_size,
          int64_t ptrs_size, int64_t num_dims,
          float ** sp_input, float ** sp_output,
          float ** sp_work,
          float *** sp_input_ptrs, float *** sp_output_ptrs,
          float *** sp_work_ptrs,  float *** sp_operator_ptrs)
  {

      // input <--> element_x
      // output <--> fx
      // work <--> element_work
      // operator_ptrs <--> operator
    bool const init = false;
    //Arrays
    fk::allocate_device(*sp_input, work_size, init);
    fk::allocate_device(*sp_output, output_size, init);
    fk::allocate_device(*sp_work, work_size, init);
    //Pointers to arrays: views
    fk::allocate_device(*sp_input_ptrs, ptrs_size, init);
    fk::allocate_device(*sp_output_ptrs, ptrs_size, init);
    fk::allocate_device(*sp_work_ptrs, ptrs_size, init);
    fk::allocate_device(*sp_operator_ptrs, ptrs_size * num_dims, init);
  }

  void deallocate_sp_space(
          float * sp_input, float * sp_output,
          float * sp_work,
          float ** sp_input_ptrs, float ** sp_output_ptrs,
          float ** sp_work_ptrs,  float ** sp_operator_ptrs)
  {
    fk::delete_device(sp_input);
    fk::delete_device(sp_output);
    fk::delete_device(sp_work);
    fk::delete_device(sp_input_ptrs);
    fk::delete_device(sp_output_ptrs);
    fk::delete_device(sp_work_ptrs);
    fk::delete_device(sp_operator_ptrs);
  }

  void convert(
          int64_t work_size, int64_t output_size,
          double * dp_input, double * dp_output,
          double * dp_work,
          float * sp_input, float * sp_output,
          float * sp_work)
  {
      for(int64_t i =0; i<work_size; i++){
          sp_input [i] = (float)  dp_input[i];
          sp_work  [i] = (float)   dp_work[i];
      }
      for(int64_t i =0; i<output_size; i++)
          sp_output[i] = (float) dp_output[i];
  }

  void convert_back(
          int64_t work_size, int64_t output_size,
          double * dp_input, double * dp_output,
          double * dp_work,
          float * sp_input, float * sp_output,
          float * sp_work)
  {
      for(int64_t i =0; i<work_size; i++){
          dp_work  [i] = (double)   sp_work[i];
          dp_input [i] = (double)  sp_input[i];
      }
      for(int64_t i =0; i<output_size; i++)
         dp_output[i] = (double) sp_output[i];
  }

// calculate how much workspace we need on device to compute a single connected
// element
//
// *does not include operator matrices - working for now on assumption they'll
// all be resident*
template<typename P>
inline double get_element_size_MB(PDE<P> const &pde)
{
  auto const elem_size      = element_segment_size(pde);
  auto const num_workspaces = 2;
  // each element requires two workspaces, X and W in Ed's code
  auto const elem_workspace_MB =
      get_MB<P>(pde.num_terms * elem_size) * num_workspaces;

  // each element requires 3 (input/output/work) + num_dims (operators) ptrs
  double const batch_size_MB = get_MB<P *>(3 + pde.num_dims);

  return elem_workspace_MB + batch_size_MB;
}

// determine how many subgrids will be required to solve the problem
// each subgrids is a subset of the element subgrid assigned to this rank,
// whose total workspace requirement is less than the limit passed in
// rank_size_MB
template<typename P>
inline int
get_num_subgrids(PDE<P> const &pde, elements::table const &elem_table,
                 element_subgrid const &grid, int const rank_size_MB)
{
  expect(grid.size() > 0);

  // determine total problem size
  auto const num_elems        = grid.size();
  double const space_per_elem = get_element_size_MB(pde);

  // determine size of assigned x and y vectors
  auto const elem_size   = element_segment_size(pde);
  auto const num_x_elems = static_cast<int64_t>(grid.nrows()) * elem_size;
  expect(num_x_elems < INT_MAX);
  auto const num_y_elems = static_cast<int64_t>(grid.ncols()) * elem_size;
  expect(num_y_elems < INT_MAX);
  double const xy_space_MB = get_MB<P>(num_y_elems + num_x_elems);

  // make sure rank size is something reasonable
  expect(space_per_elem < (0.5 * rank_size_MB));

  // size of workspaces, input, output
  double const problem_size_MB = space_per_elem * num_elems;

  // FIXME here we assume all coefficients are of equal size; if we shortcut
  // computation for identity coefficients later, we will need to do this more
  // carefully
  auto const coefficients_size_MB = static_cast<int>(std::ceil(
      get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                pde.num_terms * pde.num_dims)));

  double const table_size_MB =
      get_MB<int>(elem_table.get_active_table().size());

  // make sure the coefficient matrices/xy vectors aren't leaving us without
  // room for anything else in device workspace
  auto const remaining_rank_MB =
      rank_size_MB - coefficients_size_MB - table_size_MB - xy_space_MB;

  expect(remaining_rank_MB > space_per_elem * 4);

  // determine number of subgrids
  return static_cast<int>(std::ceil(problem_size_MB / remaining_rank_MB));
}

// helper - break subgrid into smaller subgrids to fit into DRAM
template<typename P>
inline std::vector<element_subgrid>
decompose(PDE<P> const &pde, elements::table const &elem_table,
          element_subgrid const &my_subgrid, int const workspace_size_MB)
{
  expect(workspace_size_MB > 0);

  // min number subgrids
  auto const num_subgrids =
      get_num_subgrids(pde, elem_table, my_subgrid, workspace_size_MB);

  if (num_subgrids == 1)
  {
    return std::vector<element_subgrid>{my_subgrid};
  }

  auto const max_elements_per_subgrid = my_subgrid.size() / num_subgrids;

  // max subgrid dimension (r or c)
  auto const subgrid_length =
      static_cast<int>(std::floor(std::sqrt(max_elements_per_subgrid)));

  // square tile the assigned subgrid
  std::vector<element_subgrid> grids;
  auto const round_up = [](int const to_round, int const multiple) {
    expect(multiple);
    return ((to_round + multiple - 1) / multiple) * multiple;
  };

  // create square tile iteration space
  // iterate over these creating square subgrids.
  // shrink if necessary to fit original subgrid boundary.
  auto const explode_rows = round_up(my_subgrid.nrows(), subgrid_length);
  auto const explode_cols = round_up(my_subgrid.ncols(), subgrid_length);

  for (auto i = 0; i < explode_rows / subgrid_length; ++i)
  {
    for (auto j = 0; j < explode_cols / subgrid_length; ++j)
    {
      auto const row_start = i * subgrid_length;
      auto const row_end =
          std::min(my_subgrid.row_stop, (i + 1) * subgrid_length - 1);
      auto const col_start = j * subgrid_length;
      auto const col_end =
          std::min(my_subgrid.col_stop, (j + 1) * subgrid_length - 1);
      grids.push_back(element_subgrid(row_start, row_end, col_start, col_end));
    }
  }
  return grids;
}

// single allocate kronmult workspace
template<typename P>
class kronmult_workspace
{
public:
  // private constructor, callers must enter here
  static kronmult_workspace &get_workspace(PDE<P> const &pde,
                                           elements::table const &elem_table,
                                           element_subgrid const &my_subgrid)
  {
    // function static initialization - only on first entry
    static kronmult_workspace my_workspace(pde, elem_table, my_subgrid);
    // check new values against allocated ones, resize if necessary
    my_workspace.validate(pde, my_subgrid);
    return my_workspace;
  }

  P *get_element_x() const { return element_x; }
  P *get_element_work() const { return element_work; }
  P **get_input_ptrs() const { return input_ptrs; }
  P **get_work_ptrs() const { return work_ptrs; }
  P **get_output_ptrs() const { return output_ptrs; }
  P **get_operator_ptrs() const { return operator_ptrs; }

  // no move no copy
  kronmult_workspace(kronmult_workspace const &) = delete;
  void operator=(kronmult_workspace const &)        = delete;
  kronmult_workspace(kronmult_workspace<P> &&other) = delete;
  kronmult_workspace &operator=(kronmult_workspace<P> &&other) = delete;

  ~kronmult_workspace()
  {
    fk::delete_device(element_x);
    fk::delete_device(element_work);
    fk::delete_device(input_ptrs);
    fk::delete_device(operator_ptrs);
    fk::delete_device(work_ptrs);
    fk::delete_device(output_ptrs);
  }

private:
  kronmult_workspace(PDE<P> const &pde, elements::table const &elem_table,
                     element_subgrid const &my_subgrid)
  {
    auto const degree     = pde.get_dimensions()[0].get_degree();
    auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

    workspace_size = my_subgrid.size() * deg_to_dim * pde.num_terms;
    ptrs_size      = my_subgrid.size() * pde.num_terms;

    auto const output_size = my_subgrid.nrows() * deg_to_dim;

    auto const coefficients_size_MB =
        get_MB<P>(static_cast<int64_t>(pde.get_coefficients(0, 0).size()) *
                  pde.num_terms * pde.num_dims);

    node_out() << "--- kron workspace size ---" << '\n';
    node_out() << "  coefficient size (MB): " << coefficients_size_MB << '\n';
    node_out() << "  solution vector allocation (MB): "
               << get_MB<P>(output_size) << '\n';
    node_out() << "  element table allocation (MB): "
               << get_MB<int>(elem_table.get_active_table().size()) << '\n';
    node_out() << "  workspace allocation (MB): "
               << get_MB<P>(workspace_size * 2) << "\n\n";

    // don't memset
    bool const initialize = false;
    fk::allocate_device(element_x, workspace_size, initialize);
    fk::allocate_device(element_work, workspace_size, initialize);
    fk::allocate_device(input_ptrs, ptrs_size, initialize);
    fk::allocate_device(work_ptrs, ptrs_size, initialize);
    fk::allocate_device(output_ptrs, ptrs_size, initialize);
    fk::allocate_device(operator_ptrs, ptrs_size * pde.num_dims, initialize);
  }

  void validate(PDE<P> const &pde, element_subgrid const &my_subgrid)
  {
    auto const degree     = pde.get_dimensions()[0].get_degree();
    auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));
    auto const new_workspace_size =
        my_subgrid.size() * deg_to_dim * pde.num_terms;
    auto const new_ptrs_size = my_subgrid.size() * pde.num_terms;
    if (new_workspace_size > workspace_size || new_ptrs_size > ptrs_size)
    {
      node_out() << "  reallocating kron workspace for new size!" << '\n';

      fk::delete_device(element_x);
      fk::delete_device(element_work);
      fk::delete_device(input_ptrs);
      fk::delete_device(operator_ptrs);
      fk::delete_device(work_ptrs);
      fk::delete_device(output_ptrs);

      // don't memset
      bool const initialize = false;

      // FIXME allocate once for maximum adaptivity? this would be a LOT of
      // elements, not sure we want to do that. but, the below code will crash
      // if we add so many elements that we spill out of device RAM
      fk::allocate_device(element_x, new_workspace_size, initialize);
      fk::allocate_device(element_work, new_workspace_size, initialize);
      fk::allocate_device(input_ptrs, new_ptrs_size, initialize);
      fk::allocate_device(work_ptrs, new_ptrs_size, initialize);
      fk::allocate_device(output_ptrs, new_ptrs_size, initialize);
      fk::allocate_device(operator_ptrs, new_ptrs_size * pde.num_dims,
                          initialize);

      auto const output_size = my_subgrid.nrows() * deg_to_dim;
      node_out() << "--- kron workspace resize ---" << '\n';
      node_out() << "  solution vector allocation (MB): "
                 << get_MB<P>(output_size) << '\n';
      node_out() << "  workspace allocation (MB): "
                 << get_MB<P>(new_workspace_size * 2) << "\n\n";

      workspace_size = new_workspace_size;
      ptrs_size      = new_ptrs_size;
    }
  }

  int64_t workspace_size;
  int64_t ptrs_size;

  P *element_x;
  P *element_work;
  // Array (kron batch) of arrays (element_x)
  P **input_ptrs;
  // Array (kron batch) of arrays(element_work):
  // temporary (work) memory space for kronmult computation
  P **work_ptrs;
  // Array (kron batch) of arrays(fx):
  //containing the final kronmult result
  P **output_ptrs;
  //containing the A matrix with the operators to apply to X
  // kron(A1, A2, ... Andims) \times  X
  P **operator_ptrs;
};

// private, directly execute one subgrid
template<typename P>
fk::vector<P, mem_type::view, resource::device>
execute(PDE<float> const &sp_pde,PDE<P> const &pde, elements::table const &elem_table,
        options const &program_opts, element_subgrid const &my_subgrid,
        fk::vector<P, mem_type::const_view, resource::device> const &vec_x,
        fk::vector<P, mem_type::view, resource::device> &fx, std::string previous)
{
  std::cerr << "Executing : " << previous << std::endl;
  // FIXME code relies on uniform degree across dimensions
  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  auto const output_size = my_subgrid.nrows() * deg_to_dim;
  expect(output_size == fx.size());
  auto const input_size = my_subgrid.ncols() * deg_to_dim;
  expect(input_size == vec_x.size());

  auto const &workspace =
      kronmult_workspace<P>::get_workspace(pde, elem_table, my_subgrid);

  tools::timer.start("kronmult_stage");
  // stage x vector in writable regions for each element
  auto const num_copies = my_subgrid.nrows() * pde.num_terms;
  stage_inputs_kronmult(vec_x.data(), workspace.get_element_x(), vec_x.size(),
                        num_copies);
  tools::timer.stop("kronmult_stage");

  // list building kernel needs simple arrays/pointers, can't compile our
  // objects

  // FIXME assume all operators same size - largest possible adaptivity size
  auto const lda =
      degree *
      fm::two_raised_to(
          program_opts.max_level); // leading dimension of coefficient matrices

  fk::vector<P *> const operators = [&pde, lda] {
    fk::vector<P *> builder(pde.num_terms * pde.num_dims);
    for (int i = 0; i < pde.num_terms; ++i)
    {
      for (int j = 0; j < pde.num_dims; ++j)
      {
        builder(i * pde.num_dims + j) = pde.get_coefficients(i, j).data();
        expect(pde.get_coefficients(i, j).nrows() == lda);
      }
    }
    return builder;
  }();
  fk::vector<P *, mem_type::owner, resource::device> const operators_d(
      operators.clone_onto_device());

  fk::vector<float *> const sp_operators = [&sp_pde, lda] {
    fk::vector<float *> builder(sp_pde.num_terms * sp_pde.num_dims);
    for (int i = 0; i < sp_pde.num_terms; ++i)
    {
      for (int j = 0; j < sp_pde.num_dims; ++j)
      {
        builder(i * sp_pde.num_dims + j) = sp_pde.get_coefficients(i, j).data();
        expect(sp_pde.get_coefficients(i, j).nrows() == lda);
      }
    }
    return builder;
  }();

  fk::vector<float *, mem_type::owner, resource::device> const sp_operators_d(
      sp_operators.clone_onto_device());
  // FIXME: reduce precision of workspace arrays here
  float *sp_input = NULL;
  float *sp_work = NULL;
  float *sp_output = NULL;
  float **sp_input_ptrs = NULL;
  float **sp_work_ptrs = NULL;
  float **sp_output_ptrs = NULL;
  float **sp_operator_ptrs = NULL;
  int64_t workspace_size = my_subgrid.size() * deg_to_dim * pde.num_terms;
  int64_t ptrs_size      = my_subgrid.size() * pde.num_terms;
  if constexpr (std::is_same<P, double>::value){
  allocate_sp_space( workspace_size, output_size, ptrs_size, pde.num_dims,
          &sp_input, &sp_output, &sp_work,
          &sp_input_ptrs, &sp_output_ptrs, &sp_work_ptrs, &sp_operator_ptrs);
  convert( workspace_size, output_size,
          workspace.get_element_x(), fx.data(),
          workspace.get_element_work(),
          sp_input, sp_output,
          sp_work);
  }

  // prepare lists for kronmult, on device if cuda is enabled
  tools::timer.start("kronmult_build");
  /*
   prepare_kronmult(elem_table.get_active_table().data(), operators_d.data(),
                   lda, workspace.get_element_x(), workspace.get_element_work(),
                   fx.data(), workspace.get_operator_ptrs(),
                   workspace.get_work_ptrs(), workspace.get_input_ptrs(),
                   workspace.get_output_ptrs(), degree, pde.num_terms,
  */
  prepare_kronmult<float>(elem_table.get_active_table().data(), sp_operators_d.data(),
                   lda, sp_input, sp_work,
                   sp_output,sp_operator_ptrs,
                   sp_work_ptrs,sp_input_ptrs,
                   sp_output_ptrs, degree, pde.num_terms,
                   pde.num_dims, my_subgrid.row_start, my_subgrid.row_stop,
                   my_subgrid.col_start, my_subgrid.col_stop);
  tools::timer.stop("kronmult_build");

  auto const total_kronmults = my_subgrid.size() * pde.num_terms;
  auto const flops = pde.num_dims * 2.0 * (std::pow(degree, pde.num_dims + 1)) *
                     total_kronmults;

  tools::timer.start("kronmult");
  /*
   call_kronmult(degree, workspace.get_input_ptrs(), workspace.get_output_ptrs(),
               workspace.get_work_ptrs(), workspace.get_operator_ptrs(), lda,
   */
  call_kronmult(degree, sp_input_ptrs, sp_output_ptrs,
                sp_work_ptrs, sp_operator_ptrs, lda,
                total_kronmults, pde.num_dims);
  tools::timer.stop("kronmult", flops);
  if constexpr (std::is_same<P, double>::value){
  convert_back( workspace_size, output_size,
          workspace.get_element_x(), fx.data(),
          workspace.get_element_work(),
          sp_input, sp_output,
          sp_work);
  deallocate_sp_space(sp_input, sp_output, sp_work, sp_input_ptrs,
          sp_output_ptrs, sp_work_ptrs, sp_operator_ptrs);
  }
  return fx;
}

// public, execute a given subgrid by decomposing and running over sub-subgrids
template<typename P>
fk::vector<P, mem_type::owner, resource::host>
execute(PDE<float> const &sp_pde,PDE<P> const &pde, elements::table const &elem_table,
        options const &program_opts, element_subgrid const &my_subgrid,
        int const workspace_size_MB,
        fk::vector<P, mem_type::owner, resource::host> const &x, std::string previous)
{
  auto const grids = decompose(pde, elem_table, my_subgrid, workspace_size_MB);

  auto const degree     = pde.get_dimensions()[0].get_degree();
  auto const deg_to_dim = static_cast<int>(std::pow(degree, pde.num_dims));

  auto const output_size = my_subgrid.nrows() * deg_to_dim;
  expect(output_size < INT_MAX);
  fk::vector<P, mem_type::owner, resource::device> fx_dev(output_size);
  fk::vector<P, mem_type::owner, resource::device> const x_dev(
      x.clone_onto_device());

  int cnt=0;
  for (auto const grid : grids)
  {
    auto const col_start = my_subgrid.to_local_col(grid.col_start);
    auto const col_end   = my_subgrid.to_local_col(grid.col_stop);
    auto const row_start = my_subgrid.to_local_row(grid.row_start);
    auto const row_end   = my_subgrid.to_local_row(grid.row_stop);
    fk::vector<P, mem_type::const_view, resource::device> const x_dev_grid(
        x_dev, col_start * deg_to_dim, (col_end + 1) * deg_to_dim - 1);
    fk::vector<P, mem_type::view, resource::device> fx_dev_grid(
        fx_dev, row_start * deg_to_dim, (row_end + 1) * deg_to_dim - 1);
    //std::cerr << "Grid : " << cnt << std::endl;
    cnt++;
    fx_dev_grid = kronmult::execute(sp_pde,pde, elem_table, program_opts, grid,
                                    x_dev_grid, fx_dev_grid,previous);
  }
  return fx_dev.clone_onto_host();
}

#define X(T)                                                \
  template std::vector<element_subgrid> decompose(          \
      PDE<T> const &pde, elements::table const &elem_table, \
      element_subgrid const &my_subgrid, int const workspace_size_MB);
#include "type_list_float.inc"
#undef X

#define X(T)                                                             \
  template fk::vector<T, mem_type::owner, resource::host> execute(       \
     PDE<float> const &sp_pde, PDE<T> const &pde, elements::table const &elem_table,              \
      options const &program_options, element_subgrid const &my_subgrid, \
      int const workspace_size_MB,                                       \
      fk::vector<T, mem_type::owner, resource::host> const &x, std::string );
#include "type_list_float.inc"
#undef X

} // namespace kronmult
