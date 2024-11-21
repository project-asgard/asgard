# Basic Usage

The best place to start is with the examples installed in
```
  <CMAKE_INSTRALL_PREFIX>/share/asgard/examples/
```

### Write your own PDE class

Similar to the provided examples, a PDE specification starts as a derived class
from `asgard::PDE`, where we define the operator term and sources and call the
parent `initialize()` method.
While generally speaking the two-stage initialization is an anti-pattern, here
this is done to provide maximum flexibility to the user.
We are currently exploring alternative approaches to the API but the existing
process will be supported for the foreseeable future.

In the provided example, the functions in the terms are using static members
but those use `std::function` and are very flexible, e.g., those can accept
lambda-closures.
The main reason to use static methods can variables is to avoid potential
issues that can arise from:
* capture of pointer `this` the lifetime of the object
* the relocation of the object, moving the pointer/reference
* capture by value (copy) vs. capture by reference
The approach of static variables effectively creates a singleton class and
avoids the above potential pitfalls, at the restriction of allowing the
simulation of only one PDE instance at a time.

Additional examples are provided in the source folder `src/pde/` while those
contain a lot more capabilities than shown in the simple examples, most of those
are rather cryptic and not intended for tutorial purposes.
The ASGarD team is working to improve the presentation of the examples.


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
