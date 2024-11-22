# ASGarD - Adaptive Sparse Grid Discretization

The ASGarD project has the goal of building a solver specifically targeting
high-dimensional PDEs where the "curse-of-dimensionality" has previously
precluded useful continuum / Eularian (grid or mesh based as opposed to
Monte-Carlo sampling) simulation. Our approach is based on a
Discontinuous-Galerkin finite-element solver build atop an adaptive hierarchical
sparse-grid (note this is different from the "combination technique" when applied
to sparse-grids).

To cite the ASGarD code in your work, please use:
* [![DOI](https://joss.theoj.org/papers/10.21105/joss.06766/status.svg)](https://doi.org/10.21105/joss.06766)
* [doi:10.11578/dc.20201125.5](https://www.osti.gov/doecode/biblio/48752)

Papers using ASGarD:
* [Sparse-grid Discontinuous Galerkin Methods for the Vlasov-Poisson-Lenard-Bernstein Model](https://arxiv.org/abs/2402.06493)

[Documentation of usage: https://project-asgard.github.io/asgard/](https://project-asgard.github.io/asgard/)

The [developer documentation](https://github.com/project-asgard/ASGarD/wiki/developing)
contains information about how to contribute to the ASGarD project.

## Contact Us

Issues are a great way to discuss all aspects of the ASGarD project, whether it
is to ask a general question, request a new feature, or propose a contribution
to the code base.

The ASGarD project was initiated by David Green at Oak Ridge
National Laboratory.

For technical questions, contact Miroslav Stoyanov (stoyanovmk@ornl.gov) at
Oak Ridge National Laboratory.

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
*  BLAS/LAPACK

See the detailed [Installation](doxygen/installation.md) instructions.
