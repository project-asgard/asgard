# Quick Plotting

ASGarD does not have an objective to provide plotting capabilities but rather
the capability to interpret the sparse grid storage formats and to generate
plotting data for actual plotting tools.
The plotting examples provide Python files that demonstrate the generation
of the data, both Python and HighFive must be enabled, see the installation instructions.
In addition, ASGarD provides two tools for quick plotting and prototyping:
```
  asgardplot.sh
  asgardrun.sh
```


#### Setting up the environment

Below, the assumption is that ASGarD has been installed in <prefix> in either a VENV
environment or the environment has been enabled:
```
  source <prefix>/share/asgard/asgard-env.sh
```
Then both tools will be available in the current path.
In addition, the CMake `find_package(asgard)` command can be used without the `PATH` option
which simplifies building custom PDE files.


#### The plot script

The `asgardplot.sh` script is just a shorthand for calling the Python module
as an executable, e.g.,
```
  python3 -m asgard
```
For example, running the continuity PDE, saving the file and plotting the results:
```
  <prefix>/share/asgard/pde/continuity -l 6 -t 0.5 -of cont.h5
  python3 -m asgard cont.h5
```
Alternatively
```
  <prefix>/share/asgard/pde/continuity -l 6 -t 0.5 -of cont.h5
  asgardplot.sh cont.h5
```


#### The run-and-plot script

The `asgardrun.sh` tool will run the PDE, save the output to a temporary file `_asgardplt.h5`
and then plot the file:
```
  asgardrun.sh <prefix>/share/asgard/pde/diffusion -l 5
```


#### Changing the plot view

The command plots the first two dimensions of the stored solution, extra
dimensions will be set to the middle of their min-max ranges.
Adjusting the plot can be done with the `view` option:
```
  asgardplot.sh cont.h5 -view "*:0.01"
```
The dimensions are split with `:`, the `*` indicates which dimension to plot in full,
the number is the nominal value in the other dimension.
The tool can create only one or two dimensions (one or two stars),
the rest of the dimensions must be set to a nominal value with a number.
See the \ref asgard_examples_continuity_md "continuity example" about setting a custom
view in the PDE specification.


#### Plotting to a file

The `fig` option will save a file as opposed to opening a window:
```
  asgardplot.sh cont.h5 -fig figname.png
  asgardplot.sh cont.h5 -fig figname.png -view "*:0.01"
```


#### Chaining running and plotting with options

The run script accepts `plt` options before the PDE
```
  asgardrun.sh -plt "-fig fig1.png" <prefix>/share/asgard/pde/elliptic -l 6
  asgardrun.sh -plt "-view *:0.01 -fig fig2.png" <prefix>/share/asgard/pde/elliptic -l 6
```
The quotes are needed around the `plt` option.


#### Auxiliary fields

Assuming auxiliary fields are stored in the .h5 file, e.g.,
see \ref asgard_examples_vplb "VPLB example"
```
  asgardrun.sh -plt "-aux 0" <prefix>/share/asgard/pde/vplb -m 6 -a 1.E-6 -n 0
```
The command line tool references auxiliary fields by index only,
the Python module and a custom Python script can access those by name.


#### Moments

Moments are a special type of auxiliary field and can be accessed with a dedicated command
and the associated powers:
```
  asgardrun.sh -plt "-mom 0:0" <prefix>/share/asgard/pde/bgk -dims 2 -n 1
  asgardrun.sh -plt "-mom 0:1" <prefix>/share/asgard/pde/bgk -dims 2 -n 1 -a 1.E-5
```
The moment syntax is similar to `view` but only integer powers are accepted
and only if the moment has been registered with asgard::pde_scheme::register_moment


#### Colormaps

The default [colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
used by ASGarD is `turbo` which gives good contrast from low dark-blue values to high bright-red.
However, other colormaps can be used depending on the preferences:
```
  <prefix>/share/asgard/pde/two_stream -a 1.E-4 -t 20 -of twostr.h5
  asgardplot.sh twostr.h5
  asgardplot.sh twostr.h5 -vir
  asgardplot.sh twostr.h5 -hot
  asgardplot.sh twostr.h5 -cool
  asgardplot.sh twostr.h5 -gray
  asgardplot.sh twostr.h5 -plasma
  asgardplot.sh twostr.h5 -spec
  asgardplot.sh twostr.h5 -cmap <cmap-name>
```
The `cmap` option can take any of the over 30 matplotlib maps available,
the shorthand switches are good alternatives.


#### Showing summary

Even if the Python matplotlib module is not available, file stats can be read with
```
  asgardplot.sh -s <filename>
  asgardplot.sh -stats <filename>
```


#### Additional options

For more options see:
```
  asgardrun.sh --help
  asgardplot.sh --help
  python3 -m asgard --help
```
