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

# Dependencies
*  C++17
*  cmake 3.13
*  blas

# Quickstart

Download and build
```
git clone https://github.com/project-asgard/asgard.git
cd asgard
mkdir build
cmake ../
make
ctest
./asgard
```

# Specific build instructions
## OSX (tested on Mojave)
```
git clone https://github.com/project-asgard/asgard.git
cd asgard
mkdir build
cd build
cmake ../ -DASGARD_BUILD_OPENBLAS=1
make
ctest
./asgard
```

# References

Pointers to papers, posters, and presentations about both the method and the
code go here.
