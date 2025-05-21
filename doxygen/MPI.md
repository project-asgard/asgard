# MPI - Distributed terms

ASGarD includes support for [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface)
that allows multiple connected workstations or computing nodes to simultaneously work on a problem.
In the build process, MPI is enabled with the CMake command
```
-DASGARD_USE_MPI=ON
```
On the implementation side, the MPI integration is intended to be as seamless as reasonably possible,
so that a single PDE specification can run with or without MPI on arbitrary number of ranks.

###





### Scalability

The terms defined in the pde-scheme result in sparse-Kronecker matrix-vector products,
in the code those are called kronmult operations and
even the non-linear (interpolatory) terms are implemented with 3 separate kronmults.
The distribution of the workload of a single kronmult operation is an open question;
however, multiple kronmult operations can be assigned to different MPI-nodes,
increasing the performance compared to a single node implementation.
This is advantageous, since the terms cannot be otherwise lumped together,
e.g., we cannot simply add the matrices as in other finite element methods.
Nevertheless, the current MPI strategy cannot scale to more MPI ranks
than the number of terms and will likely lose perfect scaling long before that.
