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
All options must be the same across all MPI ranks, same holds for the terms, sources and initial
conditions of the asgard::pde_scheme.

There is a number of calls that must be done simultaneously across all MPI ranks on the selected communicator
and must have the same inputs, e.g., number of time-steps, options, etc.
Those include:
* the constructor of the `asgard::discretization_manager`
* `asgard::discretization_manager::advance_time()`
* `asgard::discretization_manager::sync_mpi_state()`
* `asgard::discretization_manager::current_state_mpi()`
All fo the `_mpi()` methods can also be called without an MPI context and they will act as expected,
e.g., `sync` is a no-op and `current_state` is just the current state.`

Calling `current_state()` is normally valid only on the zero rank, as it is not normally needed by all ranks.
The `sync_mpi_state()` synchronizes the data across the ranks, `current_state_mpi()` combines calls
to sync and current-state.

The I/O methods, such save state and print stats have effect only on the leader rank,
which is rank zero on the provided MPI communicator.

### Scalability

The operator terms defined in the asgard::pde_scheme are discretized into sparse-Kronecker matrices and,
in the code, the applications of the matrix-vector operations are called kronmult operations.
Even the non-linear (interpolatory) terms are implemented with 3 separate kronmults.
The efficient distribution of the workload of a single kronmult operation is an open question,
but scalability to a few supercomputer nodes is sufficient to address real
applications of high-dimensional PDEs.


### MPI vs. OpenMP for multiple CPU threads

The OpenMP CPU muti-threading is the preferred way to use multiple threads on the same CPU.
MPI should be used in a distributed memory environment, e.g., multiple CPUs connected
through a interconnect/network.

The exception is MacOSX, which has notoriously bad support for OpenMP.
MPI is often times a better way to get performance from multiple CPU cores on an OSX machine.
