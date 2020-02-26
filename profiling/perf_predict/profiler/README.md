# profiler

This is the profiler for ASGarD. It obtains the peak memory usage and time consumption for a given PDE across a given range of levels and degrees, and produces C++ functions to predict these values.

## Requirements
---

See the requirements section for [perf_tools](../README.md)


## Usage
---


### Help

First, you should start by reading the help output.

```
usage: main.py [-h] -p PDE [-c] [-d] [-g] [-t] [-od OUT_DEGREE]
               [-lmax LEVEL_MAX] [-lmin LEVEL_MIN] [-dmax DEGREE_MAX]
               [-dmin DEGREE_MIN] [-a ASGARD_PATH] [--data-dir DATA_DIR]
               [-o OUTPUT_DIR]

Profiles memory usage for ASGarD

optional arguments:
  -h, --help            show this help message and exit
  -p PDE, --pde PDE     The PDE to profile
  -c, --clean           Don't use memory data collected by this program
                        previously
  -d, --debug           Enable debugging output
  -g, --graph           Enable graph output
  -t, --profile-time    Profile time instead of memory
  -od OUT_DEGREE, --out-degree OUT_DEGREE
                        The degree of the functions that output the a, b, and
                        c terms of the quadratic that describes memory
  -lmax LEVEL_MAX, --level-max LEVEL_MAX
                        The maximum level value to profile
  -lmin LEVEL_MIN, --level-min LEVEL_MIN
                        The minimum level value to profile
  -dmax DEGREE_MAX, --degree-max DEGREE_MAX
                        The maximum degree value to profile
  -dmin DEGREE_MIN, --degree-min DEGREE_MIN
                        The minimum degree value to profile
  -a ASGARD_PATH, --asgard-path ASGARD_PATH
                        The path to the asgard executable
  --data-dir DATA_DIR   The output dir for the data files used internally by
                        this program for generating prediction functions
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory for data readable by the user,
                        such as spreadsheets and header files
```

### In-depth meaning of flags

The typical call to the profiler looks something like this.


```bash
# in asgard/profiling/perf_tools/profiler

python main.py -p continuity_1 -lmax 8 -dmax 6 -od 7 -g -c -a ../../build/asgard
```

Thats a lot of tags. Here is the more human readable verbose version.

```bash
# in asgard/profiling/perf_tools/profiler

python main.py --pde continuity_1 --level-max 8 --degree-max 6 --out-degree 7 --graph --clean --asgard-path ../../build/asgard
```

Here's a table of the meanings of each flag.

| Flag | Meaning |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-p`, `--pde` | The name of the PDE to profile. You can only profile one PDE at a time, and the output function for the PDE cannot be used for any other PDE. Everything produced when profiling a given PDE is very specific to that PDE. |
| `-c`, `--clean` | The profiler, by default, reuses old data already obtained by the profiler. This means that when you run the profiler for the first time, the profiler will collect data on the memory usage and compute time for the given PDE, but will not recollect the data the next time you run it.  The reuse of old data is useful for when you want to regenerate the output  functions using a different polynomial degree. This flag disables the use of old data, and any data collected when using the `-c` or `--clean` will overwrite old data. |
| `-d`, `--debug` | This flag enables debug output. |
| `-g`, `--graph` | Show graphs that demonstrate the accuracy of the profile. These plots will show the actual collected data points on top of the resulting prediction function. |
| `-t`, `--profile-time` | Profile memory instead of time. You can only do one or the other at once, the profiler does not allow you to do both in one run. |
| `-od`, `--out-degree` | This flag specifies the polynomial degree of the function used to predict the memory usage or the compute time. It defaults to 4.   This number can drastically affect the accuracy of the output function. Typically, the more levels and degrees that you probe, the higher this number should be. If this number is too high for the regression to properly fit, it can negatively affect how well the resulting function can extrapolate to degrees and levels that you did not probe when profiling. |
| `-lmax`, `--level-max` | This represents the maximum level that you want to profile the PDE to.  For example, if I profile continuity_1 with the flag `-lmax 5` the profiler will profile for levels 1, 2, 3, 4, 5, and then build the prediction function with the data collected from those levels.  The profiler will not probe further than `lmax`. |
| `-lmin`, `--level-min` | This represents the minimum level that you want to profile the PDE from.  For example, if I profile continuity_2 with the flags `-lmin 3` and `-lmax 5` the profiler will profile for levels 3, 4, 5, and then build the prediction function with the data collected from those levels.  The profiler will not probe lower than `lmin`. |
| `-dmax`, `--degree-max` | This works the same as the `-lmax` flag, but specifically for the degree option for asgard. |
| `-dmin`, `--degree-min` | This works the same as the `-lmin` flag, but specifically for the degree option for asgard. |
| `-a`, `--asgard-path` | The path to the asgard executable. This value defaults to `../asgard` |
| `--data-dir` | The output directory containing the valgrind and massif output files used by the profiler |
|`-o`, `--output-dir`|The output directory containing user readable output, such as spreadsheets and header files.|