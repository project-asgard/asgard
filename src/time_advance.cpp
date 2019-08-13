#include "time_advance.hpp"
#include "element_table.hpp"
#include "fast_math.hpp"
#include "mkl.h"
#include <mkl_blas.h>
#include <mkl_rci.h>
#include <mkl_service.h>
#include <mkl_spblas.h>

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           rank_workspace<P> &rank_space,
                           std::vector<element_chunk> chunks, P const time,
                           P const dt)
{
  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);

  fm::copy(host_space.x, host_space.x_orig);
  // see
  // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
  P const a21 = 0.5;
  P const a31 = -1.0;
  P const a32 = 2.0;
  P const b1  = 1.0 / 6.0;
  P const b2  = 2.0 / 3.0;
  P const b3  = 1.0 / 6.0;
  P const c2  = 1.0 / 2.0;
  P const c3  = 1.0;

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source, time);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_1);
  P const fx_scale_1 = a21 * dt;
  fm::axpy(host_space.fx, host_space.x, fx_scale_1);

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c2 * dt);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_2);
  fm::copy(host_space.x_orig, host_space.x);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;
  fm::axpy(host_space.result_1, host_space.x, fx_scale_2a);
  fm::axpy(host_space.result_2, host_space.x, fx_scale_2b);

  apply_explicit(pde, table, chunks, host_space, rank_space);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c3 * dt);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_3);

  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::copy(host_space.x_orig, host_space.x);
  fm::axpy(host_space.result_1, host_space.x, scale_1);
  fm::axpy(host_space.result_2, host_space.x, scale_2);
  fm::axpy(host_space.result_3, host_space.x, scale_3);

  fm::copy(host_space.x, host_space.fx);
}
// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void implicit_time_advance(PDE<P> const &pde,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           explicit_system<P> &system,
                           work_set<P> const &batches, P const time, P const dt,
                           fk::matrix<P> &A)
{
  assert(system.scaled_source.size() == system.batch_input.size());
  assert(system.batch_input.size() == system.batch_output.size());
  assert(system.x_orig.size() == system.batch_input.size());
  assert(system.result_1.size() == system.batch_input.size());
  assert(system.result_2.size() == system.batch_input.size());
  assert(system.result_3.size() == system.batch_input.size());

  fm::copy(system.batch_input, system.x_orig);
  scale_sources(pde, unscaled_sources, system.scaled_source, time + dt);
  // AA = I - dt*A;
  for (int i = 0; i < A.nrows(); ++i)
  {
    for (int j = 0; j < A.ncols(); ++j)
    {
      A(i, j) = -dt * A(i, j);
    }
    A(i, i) += 1.0;
  }
  printf(" t = %.3f   dt = %.3f\n", time, dt);
  system.scaled_source.print("Scaled Sources");
  printf("===================================\n");
  A.print("AA");
  printf("===================================\n");

  system.batch_input = system.batch_input + system.scaled_source * dt;
  system.batch_input.print("batch_input");
  printf("===================================\n");
  //  system.batch_output.print("b");
  //  printf("===================================\n");
  fm::gesv(A, system.batch_input);
  system.batch_input.print("solution");
  printf("===================================\n");
  fm::copy(system.batch_input, system.batch_output);
}

template<typename P>
void implicit_iterative_time_advance(
    const PDE<P> &pde, const std::vector<fk::vector<P>> &unscaled_sources,
    explicit_system<P> &system, const work_set<P> &batches, const P time,
    const P dt, fk::matrix<P> &A)
{
  assert(system.scaled_source.size() == system.batch_input.size());
  assert(system.batch_input.size() == system.batch_output.size());
  assert(system.x_orig.size() == system.batch_input.size());
  assert(system.result_1.size() == system.batch_input.size());
  assert(system.result_2.size() == system.batch_input.size());
  assert(system.result_3.size() == system.batch_input.size());

  fm::copy(system.batch_input, system.x_orig);
  scale_sources(pde, unscaled_sources, system.scaled_source, time + dt);
  // AA = I - dt*A;
  for (int i = 0; i < A.nrows(); ++i)
  {
    for (int j = 0; j < A.ncols(); ++j)
    {
      A(i, j) = -dt * A(i, j);
    }
    A(i, i) += 1.0;
  }
  printf(" t = %.3f   dt = %.3f\n", time, dt);
  system.scaled_source.print("Scaled Sources");
  printf("===================================\n");
  A.print("AA");
  printf("===================================\n");
  system.batch_input = system.batch_input + system.scaled_source * dt;
  system.batch_input.print("batch_input");
  printf("===================================\n");
  //  system.batch_output.print("b");
  //  printf("===================================\n");

  const int par_size = 128;
  const int MAX_ITER = 20;
  MKL_INT ipar[par_size];
  P dpar[par_size];
  ipar[14] = MAX_ITER;

  const int N = system.batch_input.size();
  int n       = N;
  const int TMP_SIZE =
      ((2 * ipar[14] + 1) * n + ipar[14] * (ipar[14] + 9) / 2 + 1);
  std::vector<P> tmp(TMP_SIZE);

  MKL_INT itercount, expected_itercount = 5;
  MKL_INT RCI_request, i, ivar;
  ivar = static_cast<MKL_INT>(N);
  std::fill(system.batch_output.begin(), system.batch_output.end(), 1.0);
  const P *solution = system.batch_output.data();
  P *rhs            = system.batch_input.data();
  /*---------------------------------------------------------------------------
   * Initialize the solver
   *--------------------------------------------------------------------------*/
  dfgmres_init(&ivar, static_cast<const double *>(solution), rhs, &RCI_request,
               ipar, dpar, tmp.data());
  if (RCI_request != 0)
  {
    printf("Failed in call to dfgmres_init() : RCI_request = %d\n",
           RCI_request);
    exit(1);
  }
  printf("After dfgmres_init() : ipar[14] = %d\n", ipar[14]);

  /*---------------------------------------------------------------------------
   * Set the desired parameters:
   * LOGICAL parameters:
   * do not do the residual stopping test
   * do request for the user defined stopping test
   * do the check of the norm of the next generated vector automatically
   * DOUBLE PRECISION parameters
   * set the relative tolerance to 1.0D-3 instead of default value 1.0D-6
   *---------------------------------------------------------------------------*/
  ipar[8]  = 0;
  ipar[9]  = 1;
  ipar[11] = 1;
  dpar[0]  = 1.0E-3;
  dfgmres_check(&ivar, static_cast<const double *>(solution), rhs, &RCI_request,
                ipar, dpar, tmp.data());
  if (RCI_request != 0)
  {
    printf("Failed in call to dfgmres_check() : RCI_request = %d\n",
           RCI_request);
    exit(1);
  }
  /*---------------------------------------------------------------------------
   * Print the info about the RCI FGMRES method
   *---------------------------------------------------------------------------*/
  printf("Some info about the current run of RCI FGMRES method:\n\n");
  if (ipar[7])
  {
    printf("As ipar[7]=%d, the automatic test for the maximal number of ",
           ipar[7]);
    printf("iterations will be\nperformed\n");
  }
  else
  {
    printf("As ipar[7]=%d, the automatic test for the maximal number of ",
           ipar[7]);
    printf("iterations will be\nskipped\n");
  }
  printf("+++\n");
  if (ipar[8])
  {
    printf("As ipar[8]=%d, the automatic residual test will be performed\n",
           ipar[8]);
  }
  else
  {
    printf("As ipar[8]=%d, the automatic residual test will be skipped\n",
           ipar[8]);
  }
  printf("+++\n");
  if (ipar[9])
  {
    printf("As ipar[9]=%d, the user-defined stopping test will be ", ipar[9]);
    printf("requested via\nRCI_request=2\n");
  }
  else
  {
    printf("As ipar[9]=%d, the user-defined stopping test will not be ",
           ipar[9]);
    printf("requested, thus,\nRCI_request will not take the value 2\n");
  }
  printf("+++\n");
  if (ipar[10])
  {
    printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will be ",
           ipar[10]);
    printf("performed, thus,\nthe preconditioner action will be requested via "
           "RCI_request=3\n");
  }
  else
  {
    printf("As ipar[10]=%d, the Preconditioned FGMRES iterations will not ",
           ipar[10]);
    printf("be performed,\nthus, RCI_request will not take the value 3\n");
  }
  printf("+++\n");
  if (ipar[11])
  {
    printf("As ipar[11]=%d, the automatic test for the norm of the next ",
           ipar[11]);
    printf("generated vector is\nnot equal to zero up to rounding and ");
    printf("computational errors will be performed,\nthus, RCI_request will "
           "not take the value 4\n");
  }
  else
  {
    printf("As ipar[11]=%d, the automatic test for the norm of the next ",
           ipar[11]);
    printf("generated vector is\nnot equal to zero up to rounding and ");
    printf(
        "computational errors will be skipped,\nthus, the user-defined test ");
    printf("will be requested via RCI_request=4\n");
  }
  printf("+++\n\n");

  fm::copy(system.batch_input, system.x_orig);
  scale_sources(pde, unscaled_sources, system.scaled_source, time + dt);
  // AA = I - dt*A;
  for (int i = 0; i < A.nrows(); ++i)
  {
    for (int j = 0; j < A.ncols(); ++j)
    {
      A(i, j) = -dt * A(i, j);
    }
    A(i, i) += 1.0;
  }
  printf(" t = %.3f   dt = %.3f\n", time, dt);
  system.scaled_source.print("Scaled Sources");
  printf("===================================\n");
  A.print("AA");
  printf("===================================\n");

  system.batch_input = system.batch_input + system.scaled_source * dt;
  system.batch_input.print("batch_input");
  printf("===================================\n");
  //  system.batch_output.print("b");
  //  printf("===================================\n");
  fm::gesv(A, system.batch_input);
  system.batch_input.print("solution");
  printf("===================================\n");
  fm::copy(system.batch_input, system.batch_output);
}

// scale source vectors for time
template<typename P>
static fk::vector<P> &
scale_sources(PDE<P> const &pde,
              std::vector<fk::vector<P>> const &unscaled_sources,
              fk::vector<P> &scaled_source, P const time)
{
  // zero out final vect
  fm::scal(static_cast<P>(0.0), scaled_source);
  // scale and accumulate all sources
  for (int i = 0; i < pde.num_sources; ++i)
  {
    fm::axpy(unscaled_sources[i], scaled_source,
             pde.sources[i].time_func(time));
  }
  return scaled_source;
}

// apply the system matrix to the current solution vector using batched
// gemm (explicit time advance).
template<typename P>
static void
apply_explicit(PDE<P> const &pde, element_table const &elem_table,
               std::vector<element_chunk> const &chunks,
               host_workspace<P> &host_space, rank_workspace<P> &rank_space)
{
  fm::scal(static_cast<P>(0.0), host_space.fx);
  for (auto const &chunk : chunks)
  {
    // copy in inputs
    copy_chunk_inputs(pde, rank_space, host_space, chunk);

    // build batches for this chunk
    std::vector<batch_operands_set<P>> batches =
        build_batches(pde, elem_table, rank_space, chunk);

    // do the gemms
    P const alpha = 1.0;
    P const beta  = 0.0;
    for (int i = 0; i < pde.num_dims; ++i)
    {
      batch<P> const a = batches[i][0];
      batch<P> const b = batches[i][1];
      batch<P> const c = batches[i][2];

      batched_gemm(a, b, c, alpha, beta);
    }

    // do the reduction
    reduce_chunk(pde, rank_space, chunk);

    // copy outputs back
    copy_chunk_outputs(pde, rank_space, host_space, chunk);
  }
}

template void
explicit_time_advance(PDE<float> const &pde, element_table const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_chunk> chunks, float const time,
                      float const dt);

template void
explicit_time_advance(PDE<double> const &pde, element_table const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double> &host_space,
                      rank_workspace<double> &rank_space,
                      std::vector<element_chunk> chunks, double const time,
                      double const dt);

template void
implicit_time_advance(PDE<float> const &pde,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      explicit_system<float> &system,
                      work_set<float> const &batches, float const time,
                      float const dt, fk::matrix<float> &A);
template void
implicit_time_advance(PDE<double> const &pde,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      explicit_system<double> &system,
                      work_set<double> const &batches, double const time,
                      double const dt, fk::matrix<double> &A);

// template void
// implicit_iterative_time_advance(PDE<float> const &pde,
//                      std::vector<fk::vector<float>> const &unscaled_sources,
//                      explicit_system<float> &system,
//                      work_set<float> const &batches, float const time,
//                      float const dt, fk::matrix<float> &A);
template void implicit_iterative_time_advance(
    PDE<double> const &pde,
    std::vector<fk::vector<double>> const &unscaled_sources,
    explicit_system<double> &system, work_set<double> const &batches,
    double const time, double const dt, fk::matrix<double> &A);
