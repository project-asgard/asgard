
## unscheduled features

## v0.5.0 (9 Aug 2024)
- Rewriting local kronmult library and adding an optional global kronmult library
- Matrix-free implicit timestepping using GMRES or BICGStab solver
- IMEX time advance
- File checkpoint restart
- New PDEs

## v0.3.0 (26 Feb 2020)

- [x] multi-node capability (merged 24 Oct 2019)
    - [x] enable optional MPI build dependency
- [x] performance improvements
    - [x] batching (ptrs, parallel)
    - [x] tensors
- [x] build improvements (cray, fortran)
- [x] bugfixes

## v0.2.0 (not released)

- [x] enable profiling via CMake (merged 01 May 2019)
- [x] single gpu capability for low-level code (merged 10 Sep 2019)
    - [x] CMake CUDA language capability (merged 03 Dec 2019)
    - [x] fk::tensors understand unattached memory
    - [x] blas on single gpu
- [x] performance improvements
    - [x] forward transform
- [x] bugfixes

## v0.1.0 (20 Mar 2019)

- [x] initial structure of the setup code
    - [x] permutations, connectivity component (merged 23 Jan 2019)
    - [x] transformations component (merged 08 Feb 2019)
    - [x] missing quadrature/utilities test inputs (merged 06 Feb 2019)
    - [x] updated PDE spec for arbitrary dimension (merged 13 Feb 2019)
    - [x] operator matrices (merged 06 Mar 2019)
- [x] views for fk::tensors (merged 11 Apr 2019)
- [x] batching for fk::tensors (views) (merged 17 May 2019)
- [x] dimension agnostic kronecker products (merged 17 May 2019)
- [x] time loop (merged 22 May 2019)
