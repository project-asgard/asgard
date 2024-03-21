---
title: 'ASGarD: Adaptive Sparse Grid Discretization'
tags:
  - C++
  - Cuda
  - fusion
  - plasma
  - sparse grid
  - high dimensional
  - discontinuous Galerkin
authors:
  - name: Steven E. Hahn
    orcid: 0000-0002-2018-7904
    affiliation: 1
  - name: Miroslav K Stoyanov
    orcid: 0000-0002-8199-5577
    affiliation: 1
  - name: Stefan Schanke
    orcid: 0000-0002-1518-3538
    affiliation: 1
  - name: Eirik Endeve
    orcid: 0000-0003-1251-9507
    affiliation: 1
  - name: David L. Green
    orcid: 0000-0003-3107-1170
    affiliation: 1
  - name: Mark Cianciosa
    orcid: 0000-0001-6211-5311
    affiliation: 1
  - name: Ed D'Azevedo
    orcid: 0000-0002-6945-3206
    affiliation: 1
  - name: Wael Elwasif
    orcid: 0000-0003-0554-1036
    affiliation: 1
  - name: Coleman J. Kendrick
    orcid: 0000-0001-8808-9844
    affiliation: 1
  - name: Hao Lau
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: M. Graham Lopez
    orcid: 0000-0002-5375-2105
    affiliation: 1
  - name: Adam McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 4"
  - name: B.Tyler McDaniel
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2"
  - name: Lin Mu
    orcid: 0000-0002-2669-2696
    affiliation: "1, 3"
  - name: Timothy Younkin
    orcid: 0000-0002-7471-6840
    affiliation: 1
  - name: Hugo Brunie
    orcid: 0000-0000-0000-0000
    affiliation: "5, 6, 7"
  - name: Nestor Demeure
    orcid: 0000-0000-0000-0000
    affiliation: "5, 6"
  - name: Cory D Hauck
    orcid: 0000-0001-5559-502X
    affiliation: 1

affiliations:
 - name: Oak Ridge National Laboratory, Oak Ridge, Tennessee, USA
   index: 1
 - name: University of Tennessee Knoxville, Knoxville, Tennessee, USA
   index: 2
 - name: University of Georgia, Athens, Georgia, USA
   index: 3
 - name: South Doyle High School, Knoxville, Tennessee, USA
   index: 4
 - name: National Energy Research Scientific Computing Center, Berkeley, California, USA
   index: 5
 - name: Lawrence Berkley Laboratory, Berkeley, California, USA
   index: 6
 - name: National Institute for Research in Digital Science and Technology, France
   index: 7

date: 6 March 2024
bibliography: paper.bib
---

# Summary

Many areas of science exhibit physical [^1] processes which are described by high dimensional partial differential equations (PDEs), e.g., the 4D [@dorf2013], 5D [@candy2009] and 6D models [@juno2018] describing magnetized fusion plasmas, models describing quantum chemistry, or derivatives pricing [@bandrauk2007]. In such problems, the so called "curse of dimensionality" whereby the number of degrees of freedom (or unknowns) required to be solved for scales as $N^D$ where $N$ is the number of grid points in any given dimension $D$. A simple, albeit naive, 6D example is demonstrated in the left panel of Figure \ref{fig:scaling}. With $N=1000$ grid points in each dimension, the memory required just to store the solution vector, not to mention forming the matrix required to advance such a system in time, would exceed an exabyte - and also the available memory on the largest of supercomputers available today. The right panel of Figure \ref{fig:scaling} demonstrates potential savings for a range of problem dimensionalities and grid resolution. While there are methods to simulate such high-dimensional systems, they are mostly based on Monte-Carlo methods [@e2020] which rely on a statistical sampling such that the resulting solutions include noise. Since the noise in such methods can only be reduced at a rate proportional to $\sqrt{N_p}$ where $N_p$ is the number of Monte-Carlo samples, there is a need for continuum, or grid / mesh based methods for high-dimensional problems which both do not suffer from noise and bypass the curse of dimensionality. Here we present a simulation framework which provides such a method using adaptive sparse grids [@pfluger2010].

[^1]:This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The publisher acknowledges the US government license to provide public access under the DOE Public Access Plan (https://energy.gov/downloads/doe-public-access-plan).

The Adaptive Sparse Grid Discretization (ASGarD) code is a framework specifically targeted at solving high-dimensional PDEs using a combination of a Discontinuous-Galerkin Finite Element solver implemented atop an adaptive sparse grid basis. The adaptivity aspect allows for the sparsity of the basis to be adapted to the properties of the problem of interest, which facilitates retaining the advantages of sparse grids in cases where the standard sparse grid selection rule is not the best match. A prototype of the non-adaptive sparse-grid implementation was used to produce the results of [@dazevedo2020] for 3D time-domain Maxwell's equations. The implementation utilizes both CPU and GPU resources, as well as being single and multi-node capable. Performance portability is achieved by casting the computational kernels as linear algebra operations and relying on vendor provided BLAS libraries. Several test problems are provided, including advection up to 6D with either explicit or implicit timestepping.

![Illustration of the curse of dimensionality in the context of solving a 6 dimensional PDE (e.g., those at the heart of magnetically confined fusion plasma physics) on modern supercomputers, and how the memory required to store the solution vector (solid black curves) and the matrix (magenta curves) in both naive and Sparse Grid based discretizations as the resolution of the simulation domain is varied. Memory limits of the Titan and Summit supercomputers at Oak Ridge National Laboratory, in addition to an approximate value for an ExaScale supercomputer, are overlaid for context](figures/sparse-vs-full.png)

![Potential memory savings of a Sparse Grid based solver represented as the ratio of the naive tensor product full-grid (FG) degrees of freedom (DoF) to the sparse-grid (SG) DoF.](figures/savings.png) 

# Statement of Need

The goal of ASGarD is to facilitate and promote the use of adaptive sparse-grid methods by domain scientists for the approximation of kinetic models by providing a robust yet flexible adaptive sparse-grid library.


# Acknowledgements

This research used resources of the Oak Ridge Leadership Computing Facility (OLCF) at the Oak Ridge National Laboratory, and the National Energy Research Scientific Computing Center (NERSC), which are supported by the Office of Science of the U.S. Department of Energy under Contract Numbers DE-AC05-00OR22725 and DE-AC02-05CH11231 respectively.

# References
