# Quick Plotting

ASGarD does not have an objective to provide plotting capabilities but rather
the capability to interpret the sparse grid storage formats and to generate
plotting data for actual plotting tools.
After enabling both Python and HighFive (see the installation instruction),
the asgard python module can be used as an executable:

Get the version, i.e., plot nothing and show only the library build information:
```
  python3 -m asgard
```

Quick plot command of the first two dimensions of a stored solution, extra
dimensions will be set to the middle of their min-max ranges:
```
  python3 -m asgard outfile.h5
```
The quick plot command will use matplotlib and make a basic image (or 1D curve).
The installed examples show how to obtain the raw data and enable fine grained
control over the plotting format or even use a matplotlib alternatives.

If matplotlib plot can also be written to an image file, e.g., if the data files
are stored on a remote machine that has matplotlib but no display connection:
```
  python3 -m asgard outfile.h5 outfile.png
```
Here, `outfile.png` is any supported matplotlib format.

If matplotlib is missing or we want to skip plotting, we can print only the
file high-level meta data to the console:
```
  python3 -m asgard -s outfile.h5
```
The `-s` switch can be replaced with either `-stats` or `-summary`.




