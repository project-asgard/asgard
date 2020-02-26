# perf_predict

perf_predict is a tool that profiles ASGarD and builds a header file to predict

- Memory usage
- Compute time

## Requirements

---

Python3

To install the dependencies, run the following command.

```bash
# in asgard/profiling
python3 -m pip install -r perf_predict/requirements.txt
```

## Usage

---

perf_predict takes one positional argument and another argument of variable size: the path to the asgard executable, and the PDEs to reprofile. These arguments are manipulated and passed to [the profiler](./profiler/README.md) and [the builder](./builder/README.md) to profile and build the output header file.

perf_predict will only profile the PDEs it is told to. If I run the following command, perf_predict will only profile `continuity_1`.

```bash
# in asgard/profiling
python perf_predict/main.py ../build/asgard continuity_1
```

If I then run this command, perf_predict will only profile `continuity_2`. However, it will reuse the old profile generated for `continuity_1` from the last command when building the output header file. This way, you don't have to reprofile each PDE when you only want to profile one.

```bash
# in asgard/profiling
python perf_predict/main.py ../build/asgard continuity_2
```

I can also profile several PDEs at once. The following command will reprofile `continuity_1`, and will profile `continuity_3` and `vlasov4` for the first time because I have not executed perf_predict on these two PDEs before.

```bash
# in asgard/profiling
python perf_predict/main.py ../build/asgard vlasov4 continuity_1 continuity_3
```

If you try to predict the memory usage or compute time for a PDE that has not been profiled by perf_predict, meaning that you havent run the command:

```bash
# in asgard/profiling
python perf_predict/main.py ../build/asgard your_pde_name
```

then `intermediate_mem_usage` and the `expected_time` functions will return zero.

### New PDEs

The only requirement for profiling new PDEs is to add them to the `PDE_opts` enum in pde.hpp, which you would have to do anyways. Other than that, no extra work is required. This is because perf_predict only passes the name of the PDE to asgard when profiling.

### Output

perf_predict outputs `CSV` files (excel spreadsheets) showing output data, graphs, and of course, the output header file.

### Spreadsheets

To view the `CSV` files, run the following commands.

```bash
# in asgard/profiling

# Any reprofiling will remake old CSVs, essentially you can run any perf_predict command here
python perf_predict/main.py ../build/asgard continuity_1 continuity_2 continuity_3

# perf_predict will place all output files in a folder named `output` in your current directory
ls ./output/csv
```

### Graphs

To view some graphs, run the following commands.

```bash
# in asgard/profiling

# The -g tag will show the graphs representing the generated prediction function versus the actual data points collected
# The following command only shows the graphs for continuity_1
python perf_predict/main.py ../build/asgard continuity_1 -g

# The following command shows the graphs for continuity_1 and continuity_2, in that order
python perf_predict/main.py ../build/asgard continuity_1 continuity_2 -g
```

### Output functions

You can view the generated prediction functions in the `output` directory in the `mem_pred` and `time_pred` folder.

Headers in the `mem_pred` folder predict memory usage for a given PDE, and headers in the `time_pred` folder predict compute time for a given PDE.

Each header contains a single C++ function named appropriately after the PDE.

Here's an example output header.

```c++
// in output/mem_pred/continuity_1.hpp

double continuity_1_MB(int level, int degree)
{
    level -= 1;
    degree -= 2;

    double a = 0.0006950717377937755 * pow(level, 0) + 0.08180355194804746 * pow(level, 1) + -0.14893654220779537 * pow(level, 2) + 0.09175121753246997 * pow(level, 3) + -0.023113050865801326 * pow(level, 4) + 0.0021498928571428854 * pow(level, 5);
    double b = -0.0041243189857484595 * pow(level, 0) + 0.2918814170562537 * pow(level, 1) + -0.5370717417748929 * pow(level, 2) + 0.33645721017316355 * pow(level, 3) + -0.08588484350649399 * pow(level, 4) + 0.008086705238095251 * pow(level, 5);
    double c = 0.2728995564626067 * pow(level, 0) + 0.30457536571426647 * pow(level, 1) + -0.5703898238095324 * pow(level, 2) + 0.3522967380952452 * pow(level, 3) + -0.08863227142857273 * pow(level, 4) + 0.00823663904761912 * pow(level, 5);

    return a * pow(degree, 2) + b * degree + c;
}
```

### The Output Header

The output header contains all the prediction functions for the previously profiled PDEs, along with two extra functions.

Here are their declarations.

```c++
// The function that gives the predicted mem usage for a given PDE at a level and degree
double intermediate_mem_usage(PDE_opts pde, int level, int degree);

// The function that gives the predicted compute time for a given PDE at a level and degree
double expected_time(PDE_opts pde, int level, int degree);
```

## TODO

---

- [x] Add READMEs for the profiler and the builder programs
- [x] Make `profiler/spreedsheet.MemorySheet` and `profiler/spreedsheet.TimeSheet` expand with the size of their data instead of having a predetermined and fixed size.
- [x] Fix misnomer in profiler, `predicted` memory usage is actually the known allocated `Workspace` memory usage output by `asgard/main.cpp`
