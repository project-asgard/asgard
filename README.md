# ASGarD - Adaptive Sparse Grid Discretization

To cite the ASGarD code in your work, please use: (TODO)

The ASGarD project has the goal of building an solver specifically targeting
high-dimensional PDEs where the "curse-of-dimensionality" has previously
precluded useful continuum / Eularian (grid or mesh based as opposed to
Monte-Carlo sampling) simulation. Our approach is based on a
Discontinuous-Galerkin finite-element solver build atop an adaptive hierarchical
sparse-grid (note this is different from the "combination tecnique" when applied
to sparse-grids).

The [developer documentation](https://github.com/project-asgard/ASGarD/wiki/developing)
contains information about how to contribute to the ASGarD project.

- (TODO) user docs about building/using the code
- (TODO) docs about the method

## Contact Us

Issues are a great way to discuss all aspects of the ASGarD project, whether it
is to ask a general question, request a new feature, or propose a contribution
to the code base.

The ASGarD project is led by David Green (greendl1@ornl.gov) at Oak Ridge
National Laboratory.

## Automated Test Status

| Test  | Status (Develop) |
| ----- | ---------------- |
| format/clang   | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiK0RaTVI5UGNoY2k2d09KVXZOL2F5VXExQ0kxUkJVWWdLY1hKRHN0TUV1SXZHMXdUUGFYbmljUXFHd3YwRjR2REVFb01iMENhUmhRSFg3YUFTK21SQlowPSIsIml2UGFyYW1ldGVyU3BlYyI6ImJwQVg1RlEvT0ZyUzNUeFYiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| warnings/clang   | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiR1Jjc3ZhUjZuTEtjUUlVMUVQTkZJcjEyVEFGQnJvTmJtT2dhbUpsNldyZjJwc3Y4bGZDeU92dUZGY2kxK0RFREwzS2NCMkUrVHZobGVOQU1IYmlYWTBzPSIsIml2UGFyYW1ldGVyU3BlYyI6IlpENzRoemxXRkNXSkdoek0iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| unit/g++       | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiRUg1VlpxTm4yTWh1QndocTUxNGx6UXp1R3VGZ3d4dkN2eEtMRlVEMzVJWDBXTFFEamlnRVJlMUFJcG41dmFndm9sNi9uKzlGSVRBNnRWU1laWGlieG1NPSIsIml2UGFyYW1ldGVyU3BlYyI6IjRXdE8xTXBxT3hSREd6VW4iLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| unit/clang++   | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiTmRsQVFTa0YwUkplOFZmaWIzV1lkK0hBei8rUDhGMTFaZ2dOZnpwT2FTU3l6VzF4L2NvM0NZSWJlUHZmZnpZVURzSDVTejR3SWFqNlRZMmlIY25EMWNBPSIsIml2UGFyYW1ldGVyU3BlYyI6IjBjd1pWaG5DWWFWK0YraVkiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| unit/g++/mpi   | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiRDFrWWlURW5EZEU1TC8rSmlIRXkzdkxMbTUxZFRPUG9FYUlFbTBGRFJZVmlWdi9yMUlUZkloSXVGTWtNaFEwMDRJU3JhTGVQYnZsMlFLVkJRNFdVNFZVPSIsIml2UGFyYW1ldGVyU3BlYyI6InlucmhYb1plajlSWkd1YlEiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| unit/g++/cuda  | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiblkzVDBCNm95TkdzMTlRUzRGbU9SVm5SMlNTVjR2amQySG1jQ0cwNnZjQlBnbklvOGhBRzhaOUpLK3pHNjZYKzhsU1M2amR6OUkyQ2lCTWZuWGY5UTlnPSIsIml2UGFyYW1ldGVyU3BlYyI6Ijd2QSsxWmJRem9UTXgwQXIiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |
| unit/g++/io    | ![Build Status](https://codebuild.us-east-2.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiMDVGaGxuelZDVlF3SHY5ckJsOXJ3ejBIOGVpQ21Kd29aRmF6VENHeFdEMnhUbFlpeXdXc216YXU0NnFQV08zdHoxTDhCTG14bWVmU1BsSm1zZzlSZkJjPSIsIml2UGFyYW1ldGVyU3BlYyI6Ii82OEtsNnlkQTZ1TGdRVWwiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=develop) |

# Dependencies

*  C++17
*  cmake 3.19
*  blas

# Optional depedencies

* cuda
* mpi
* highfive/hdf5
* MATLAB
* ScaLAPACK

# Quickstart

Download and build
```
git clone https://github.com/project-asgard/asgard.git
cd asgard
mkdir build && cd build
cmake ../
make
ctest
./asgard
```

For best performance (especially on accelerators) please pass `-DCMAKE_BUILD_TYPE=Release` to disable asserts when building the code.

To see a list of available PDEs, run `./asgard --available_pdes`. The listed PDEs can be selected using the `-p` argument to asgard.

To see the list of all runtime options, run `./asgard --help`.

For specific platform build instructions, [see this wiki page.](https://github.com/project-asgard/asgard/wiki/platforms)

## MATLAB Interface 

Asgard can interface with MATLAB to plot, share data, and run scripts. To enable this, compile with `-DASGARD_USE_MATLAB=ON`. 

If using only for plotting purposes, then Asgard can be run without any additional arguments. However, for 
sharing data with MATLAB, Asgard will need to connect to a shared session. This can be done in MATLAB by 
running
 ```
 matlab.engine.shareEngine
 ```

Asgard should automatically connect with this shared session, but the engine name can also be specified
directly by giving the result of
```
matlab.engine.engineName
```
to Asgard with the `--matlab_name` option.

Note: do not start MATLAB in the DG-SparseGrid repo when trying to use this for plotting.

## ScaLAPACK Integration

The ASGARD\_USE\_SCALAPACK build option adds the "scalapack" solver option to ASGarD. In the implicit timestep method, this option
distributes the matrix solve across multiple processes and nodes and enables larger problems that are too large for a single computer.

This build option is only available when ASGARD\_USE\_MPI=ON. Unfortunately, setting ASGARD\_USE\_SCALAPACK alone is not sufficient
and doesn't change the ASGARD\_USE\_MPI setting.
