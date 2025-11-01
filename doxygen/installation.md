# Installation

[TOC]

### Document Sections
* Requirements
* Installation
* Basic usage


### Requirements

Minimum requirements run ASGarD
* a C/C++ compiler with support for C++-17
* [CMake](https://cmake.org/) build system
* [Basic Linear Algebra Subroutine (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [Linear Algebra PACKage (LAPACK)](http://www.netlib.org/lapack/)
    * many optimized BLAS and LAPACK implementations exist, e.g., OpenBLAS, MKL, Blis/Flame

Recommended but optional
* [OpenMP](https://en.wikipedia.org/wiki/OpenMP) for CPU multi-threading and better vectorization using AVX and AVX512 registers
    * supported by GCC and most recent versions of Clang (16 or newer)
* If you have Nvidia GPU ASGarD can take advantage of the [linear algebra libraries](https://developer.nvidia.com/cublas) and custom [CUDA kernels](https://developer.nvidia.com/cuda-zone)
* If you have AMD GPU compatible with the AMD ROCm framework, ASGarD can take advantage of the [linear algebra libraries](https://rocm.docs.amd.com/en/latest/reference/api-libraries.html) and custom [HIP kernels](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/what_is_hip.html)
* [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and [HighFive](https://bluebrain.github.io/HighFive/) libraries to output the solution state
* Python bindings using [h5py](https://www.h5py.org/) and [numpy](https://numpy.org/) for easier visualization and HDF5 post-processing
* [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) capabilities for spreading the workload across multiple computing nodes

Other CMake options
* dynamic/shared libraries are used by default, static build is possible with `-DBUILD_SHARED_LIBS=OFF`
    * Python bindings require shared libraries
* tests with CMake's ctest are enabled by default, disable with `-DASGARD_BUILD_TESTS=OFF`
* ASGarD builds with both single (float) and double precision, pick just one for faster compile time

ASGarD has the ability to automatically download and install OpenBLAS and HDF5.
However, it is recommended to use system provided libraries, available in most Linux distributions and HPC systems.


### Installation

CMake uses out-of-source build, clone the repo and build in a subfolder
```
  git clone https://github.com/project-asgard/asgard.git
  cd asgard
  mkdir build
  cd build
  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=../install \
    -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
    -D ASGARD_USE_OPENMP=ON \
    -D ASGARD_USE_PYTHON=ON \
    -D ASGARD_USE_HIGHFIVE=ON \
    -D ASGARD_PRECISIONS=double \
    ..
  cmake --build . -j
  ctest
  cmake install .
```

On a OSX system with newer Apple M chips,
the builtin BLAS/LAPACK library requires an additional flag
```
    -D CMAKE_CXX_FLAGS="-march=native -mtune=native -DACCELERATE_NEW_LAPACK"
```

The installation step is required to use ASGarD as a library for an external project,
allowing the user to define their own PDE specification without intruding into the ASGarD code.
However, the currently available equations and all tools can be used directly from the
build folder.

It is recommended to use a dedicated `CMAKE_INSTALL_PREFIX` as opposed to common
locations such as `~/.local/` or `/opt/`, which will make it easier to manage
different versions and options, e.g., CPU vs. GPU,
as well as uninstall everything by deleting the folder.

### CMake options

| option | action |
| ---- | ---- |
| ASGARD_USE_OPENMP               | Enable OpenMP                                               |
| ASGARD_TEST_OMP_NUM_THREADS     | Set the number of threads to use for testing (only testing) |
| ASGARD_USE_CUDA                 | Enable Nvidia CUDA                                          |
| ASGARD_USE_ROCM                 | Enable AMD ROCm                                             |
| ASGARD_GPU_MEMGREEDY            | Select the GPU algorithm that uses more memory (default on) |
| ASGARD_USE_MPI                  | Enable MPI distributed computing                            |
| ASGARD_MPI_BCAST_THRESHOLD      | MPI: number of ranks to switch from Send/Recv to Bcast      |
| ASGARD_MPI_REDUCE_THRESHOLD     | MPI: number of ranks to switch from Send/Recv to Reduce     |
| ASGARD_TEST_MPI_OMP_NUM_THREADS | MPI: number of OpenMP threads to use during MPI testing     |
| ASGARD_USE_PYTHON               | Enable the Python bindings, requires HighFive               |
| ASGARD_USE_HIGHFIVE             | Enable the HDF5/HighFive I/O support                        |
| ASGARD_BUILD_OPENBLAS           | Compile OpenBLAS together with ASGarD                       |
| ASGARD_BUILD_HDF5               | Compile HDF5 together with ASGarD                           |
| ASGARD_PRECISIONS               | Should be "double", "float" or "double;float"               |
| ASGARD_USE_TIMER                | Record timing of various stages of the simulation           |
| ASGARD_USE_FLOPCOUNTER          | Count flops per second for Kronecker matrix-vector products |
| ASGARD_RECOMMENDED_DEFAULTS     | Enables OpenMP and some flags (see below)                   |

Notes:
* OpenMP is not supported on Apple OSX platforms.
* CUDA and ROCM cannot be enabled simultaneously.
* The default greedy algorithm for GPU calculations uses more memory by explicitly indexing
  the connections between sparse-grid cells, which is faster (especially for smaller problems)
  while it uses lots of more memory (especially for large problems).
* The best values for `MPI_BCAST` and `MPI_REDUCE` thresholds are system specific,
  several tests may be needed to find the best values for a specific platform.
* `ASGARD_TEST_MPI_OMP_NUM_THREADS` affects only some of the tests when both MPI and OpenMP are enabled,
  the option is there to avoid oversubscribing the CPU which will result in a massive drop of performance.
* It is recommended to use system native BLAS and HDF5, the build options are provided
  for convince and mostly used in the testing containers.
* Use `double` precision when developing a new PDE scheme and switch to `float` only
  when sure that the lower precision is enough.
* The timer comes with very low overhead and it is useful for debugging, but it is probably
  superfluous for a long run.
* The flop counter comes with significant overhead, use it only for profiling and debugging.
* ASGarD makes a heavy use of both OpenMP multi-threading and SIMD features, and while SIMD is not
  supported under Clang, it is recommended to enable flags:
```
  -mfma            enable fused-multiply-add instructions
  -march=native    auto-detects CPU features
  -mtune=native    auto-detects CPU features
  -mavx512f        if the CPU supports AVX512
  -mavx512dq       if the CPU supports AVX512
```

### Python pip install

ASGarD has a pip installer called [ornl-asgard](https://pypi.org/project/ornl-asgard/) to avoid conflicts with existing project name.

Both venv and user-space installs are supported, e.g.,
```
python3 -m pip install ornl-asgard --user
```
The `--user` tag is not needed for a venv install
```
python3 -m pip install ornl-asgard
```

Some systems, and especially OSX, often come with different versions of python,
be consistent between the version used for install and runtime.

MPI and GPU support can be enabled using the pip-installer by setting an environment variable, e.g.,
```
export ASGARD_USE_MPI=ON
export ASGARD_USE_CUDA=ON
python3 -m pip install ornl-asgard
```
However, in this mode, CMake must be able to automatically find all necessary components,
i.e., it is not possible to pass fine-grained details about the path to MPI, ROCm or CUDA
using the PyPIP installer, use CMake instead.

The ASGarD executable and python module provide build option summary:
```
  asgard -v
  python3 -m asgard -v
```

### Basic usage

Check out the installed examples in
```
  <CMAKE_INSTALL_PREFIX>/share/asgard/examples
```
The prefix used by the pip-installer is either the root of the venv environment
or `.local` subfolder for the user home folder.

Using the pip-installer, so long as the venv is activated, the library, headers and python module
are in the environment path. When using a CMake installer, the library and the optional python
module can be loaded with
```
  source <CMAKE_INSTALL_PREFIX>/share/asgard/asgard-env.sh
```

For more details, see the [Basic Usage Section.](basic_usage.md)
