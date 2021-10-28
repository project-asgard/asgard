---
title: 'ASGarD: Adaptive Sparse Grid Discretization'
tags:
  - C++
  - fusion
  - plasma
  - advection
  - high dimensional
authors:
  - name: Ed D'Azevedo
    orcid: 0000-0002-6945-3206
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Wael Elwasif
    orcid: 0000-0003-0554-1036
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: David L. Green
    orcid: 0000-0003-3107-1170
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Hao Lau
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Graham Lopez
    orcid: 0000-0002-5375-2105
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Adam McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 4" # (Multiple affiliations must be quoted)
  - name: Benjamin T. McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Lin Mu
    orcid: 0000-0002-2669-2696
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Timothy Younkin
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Steven E. Hahn
    orcid: 0000-0002-2018-7904
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Coleman J. Kendrick
    orcid: 0000-0001-8808-9844
    affiliation: 1 # (Multiple affiliations must be quoted)

affiliations:
 - name: Oak Ridge National Laboratory
   index: 1
 - name: University of Tennessee Knoxville
   index: 2
 - name: University of Georgia
   index: 3
 - name: South Doyle High School, Knoxville, TN
   index: 4
date: 28 October 2021

bibliography: paper.bib
---
# Statement of Need

Many areas of science exhibit physical processes which are well described by high dimensional partial differential equations (PDEs), e.g., the 4D, 5D and 6D models describing magnetized fusion plasmas [@Juno:2017], or the ... models describing quantum mechanical interactions of several bodies [@...]. In such problems, the so called "curse of dimensionality" whereby the number of degrees of freedom (or unknowns) required to be solved for scales as $N^D$ where $N$ is the number of grid points in any given dimension. A simple, albeit naive, 6D example with $N=1000$ grid points in each dimension would require more than an exabyte of memory just for store the solution vector, not to mention forming the matrix required to advance such a system in time. While there are methods to simulate such high-dimensional systems, they are mostly based on Monte-Carlo methods which are based on a statistical sampling such that the resulting solutions are noisy. Since the noise in such methods can only be reduced at a rate proportional to $\sqrt{N_p}$ where $N_p$ is the number of Monte-Carlo samples, there is a need for continuum, or grid / mesh based methods for high-dimensional problems which both do not suffer from noise, but which additionally bypass the curse of dimenstionality. Here we present a simulation framework which provides such a method. 

# Summary

# Mathematics

# Citations

# Acknowledgements

This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility operated under Contract No. DE-AC02-05CH11231.

# References
