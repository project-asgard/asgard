# MPI - distributed terms

ASGarD includes support for [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface)
that allows multiple connected workstations or computing nodes to simultaneously work on a problem.
In the build process, MPI is enabled with the CMake command
```
-DASGARD_USE_MPI=ON
```
On the implementation side, the MPI integration is intended to be as seamless as reasonably possible,
so that a single PDE specification can run with or without MPI on arbitrary number of ranks.


### Code changes

Using MPI requires a call to `MPI_Init()` (and finalize), this can be done manually
or using a RAII library init object.
```
int main(int argc, char **argv)
{
  asgard::libasgard_runtime running_(argc, argv);
```
The `asgard::libasgard_runtime` should not be used with manual calls to `MPI_Init()`
and the ASGarD library does not require initialization.

The `asgard::prog_opts` object has a communicator operation, available only
in conjunction with MPI
```
  asgard::prog_opts options(argc, argv);
  #ifdef ASGARD_USE_MPI
  options.mpicomm = comm;
  #endif
```
The default communicator is `MPI_COMM_WORLD`.

All MPI ranks (on the selected communicator) must simultaneously called together by all the ranks.
Same holds for `advance_time()` and the number of steps must be the same too.

Calling `current_state()` is valid only on the zero rank,
if the state is needed across all ranks, use `current_state_mpi()`.
The `_mpi()` method can be used even without MPI, it will simply yield to the no-MPI option.


### Scalability

The operator terms defined in the asgard::pde_scheme are discretized into sparse-Kronecker matrices and,
in the code, the applications of the matrix-vector operations are called kronmult operations.
Even the non-linear (interpolatory) terms are implemented with 3 separate kronmults.
The efficient distribution of the workload of a single kronmult operation is an open question;
however, multiple kronmult operations can be assigned to different MPI-ranks,
increasing the performance compared to a single node implementation.
This is advantageous, since the terms cannot be otherwise lumped together,
e.g., we cannot simply add the matrices as in other common finite element methods.
Unfortunately, this limits the scalability of ASGarD and
the current MPI strategy cannot scale to more MPI ranks
than the number of terms and will see diminishing returns long before that.

Nevertheless, the overall performance of the library is sufficient to address real
applications of high-dimensional PDEs, even using a single high end workstation.


### MPI vs. OpenMP for multiple CPU threads

The OpenMP CPU muti-threading that uses shared-memory paradigm is much better way
to utilize multiple CPU cores, compared to the distributed memory framework of MPI.
A single workstation should always use OpenMP, which can also distribute the workload
of a single kronmult operation.
Two workstations should use two MPI ranks and multiple OpenMP threads.

The exception to this rule is MacOSX, which has notoriously bad OpenMP support.
Using MPI on a single OSX device will likely yield better performance,
despite the limitations of MPI scalability.
