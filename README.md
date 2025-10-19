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

[![Ubuntu-tested](https://github.com/project-asgard/asgard/actions/workflows/build-ubuntu.yaml/badge.svg?branch=develop)](https://github.com/project-asgard/asgard/actions/workflows/build-ubuntu.yaml)
[![MacOSX-tested](https://github.com/project-asgard/asgard/actions/workflows/build-macos.yaml/badge.svg?branch=develop)](https://github.com/project-asgard/asgard/actions/workflows/build-macos.yaml)


# Dependencies

*  C++17
*  cmake 3.19
*  BLAS/LAPACK

See the detailed [Installation](doxygen/installation.md) instructions.
