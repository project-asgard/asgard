# Basic Usage

The best place to start is with the examples installed in
```
  <CMAKE_INSTRALL_PREFIX>/share/asgard/examples/
```
The prefix used by the pip-installer is either the root of the venv environment
or `.local` subfolder for the user home folder.

### Write your own PDE scheme

The first step is create a program options `asgard::prog_opts` object that
describes sparse grid level, polynomial degree, time-stepping parameters,
and so on.
The options can be set manually by the user, or read from the command line
or input file, where ASGarD provides a convenient way to parce the command
line parameters and potentially add new options.

The second step is to create an `asgard::pde_domain` object defining the
number of dimensions and upper/lower limit for each direction.

Then the `asgard::pde_scheme` object is created, the domain and options are
either copied or moved inside, and can no longer be changed.
Then the scheme has to be populated with the operator terms, source terms,
initial and boundary conditions.

The final step is to copy (or move) the PDE scheme into an
`asgard::discretization_manager`, which internally controls the operator
discretization and performs time-stepping, plotting I/O, error checking, etc.

Really, read the examples!

### Compile against the installed libasgard

The examples also include a simple `CMakeLists.txt` file that shows how to
link to ASGarD. The key command is:
```
find_package(asgard)
```
The example CMake file contains the installation path which can be hardcoded
into the custom project. Alternatively, the `PATHS` directive can be omitted
but then we have to provide `asgard_ROOT` to the CMake configuration, as per
the established CMake conventions.

As an alternative, there environment setup script:
```
  source <CMAKE_INSTRALL_PREFIX>/share/asgard/asgard-env.sh
```
Which will set `asgard_ROOT`, the `PATH` to the asgard executable and the path
to the python module.


### Running and plotting

The current way to plot and post-process the solution to a PDE is to first run
the problem and output an HDF5 file. Then load the file using python and the
provided asgard module or use the Python-MATLAB integration. This requires CMake
options
```
  -D ASGARD_USE_PYTHON=ON
  -D ASGARD_USE_HIGHFIVE=ON
```

The data stored in the HDF5 file is in sparse grid hierarchical format, full
reconstruction over a dense high-dimensional domain is computationally
impractical to infeasible. The asgard python module links to the C++ code from
libasgard and allows for fast reconstruction of 1D and 2D slices of the domain,
or reconstruction at arbitrary set of points.

Python must be able to find the files for the asgard module, which an be done
in several ways:
* source the `asgard-env.sh` setup script, see above
* manually set `PYTHONPATH` environment variable
* add the path to the python script directly
```
  import sys
  sys.path.append('<CMAKE_INSTRALL_PREFIX>/lib/python<version>/site-packages/')
  # using the ASGarD install prefix and python version
```
* setup a [python venv](https://docs.python.org/3/library/venv.html) and install
  ASGarD into the same python folder, i.e.,
```
  python3 -m venv <path-to-venv>
  source <path-to-venv>/bin/activate
  ...
  cmake -DCMAKE_INSTALL_PREFIX=<path-to-venv> ....
```
See the [Quick Plotting Documentation.](plotting.md)

The installed examples contain detailed Python and MATLAB demonstration scripts.
