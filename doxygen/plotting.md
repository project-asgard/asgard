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
Below, the assumption is that ASGarD has been installed in <prefix> in either a VENV
environment or the environment has been enabled:
```
  source <prefix>/share/asgard/asgard-env.sh
```
Then both tools will be available in the current path.

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

The `asgardrun.sh` tool will run the PDE, save the output to a temporary file `_asgardplt.h5`
and then plot the file:
```
  asgardrun.sh <prefix>/share/asgard/pde/diffusion -l 5
```



If matplotlib plot can also be written to an image file, e.g., if the data files
are stored on a remote machine that has matplotlib but no display connection:
```
  python3 -m asgard outfile.h5 -fig outfile.png
```
Here, `outfile.png` is any supported matplotlib format.

If matplotlib is missing or we want to skip plotting, we can print only the
file high-level meta data to the console:
```
  python3 -m asgard -s outfile.h5
```
The `-s` switch can be replaced with either `-stats` or `-summary`.

For more options see:
```
  asgardrun.sh --help
  asgardplot.sh --help
  python3 -m asgard --help
```

Even if the Python matplotlib module is not available, file stats can be read with
```
  asgardplot.sh -s <filename>
```

