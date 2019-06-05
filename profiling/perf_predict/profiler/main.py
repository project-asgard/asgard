from graph import Graph
from git import get_commit_hash, get_commit_date
from profile import profile_mem_usage, profile_time
from fileio import write_mem, write_time
from options import DEBUGGING, OUTPUT_FUNCTION_DEGREE, LEVELS, GRAPHING, CSV_OUTPUT_DIR, PDE, PROFILING_TIME


def main():
    if PROFILING_TIME:
        generate_time_predictor()
    else:
        generate_mem_predictor()


# Profile time and output resulting header files, csv spreadsheets, and graphs
def generate_time_predictor():
    # Get expressions for a, b, and c terms, and the TimeSheet object
    a_fn, b_fn, c_fn, time_sheet = profile_time(
        OUTPUT_FUNCTION_DEGREE, OUTPUT_FUNCTION_DEGREE, OUTPUT_FUNCTION_DEGREE
    )
    print(f'Operation Successful')

    # This is functionally identical the C++ output function,
    # this is used to test the C++ output.
    # (Im currying this to add to the graph for each level)
    def f(level): return lambda degree: a_fn(level) * \
        degree ** 2 + b_fn(level) * degree + c_fn(level)

    # Print tested outputs for the resulting function
    if DEBUGGING:
        for level in range(1, 6):
            for degree in range(2, 7):
                debug(f'level={level} degree={degree}: {f(level-1)(degree-2)}')

    # Write C++ code for time prediction to file
    write_time(a_fn, b_fn, c_fn, get_commit_hash(PDE), get_commit_date(PDE))

    open(f'{CSV_OUTPUT_DIR}/{PDE}_time.csv', 'w').write(str(time_sheet))
    print(f'Wrote output time spreadsheet to {CSV_OUTPUT_DIR}/{PDE}_time.csv')

    # Show graphical output if requested by user
    graph(a_fn, b_fn, c_fn)


# Profile memory and output resulting header files, csv spreadsheets, and graphs
def generate_mem_predictor():
    # Get expressions for a, b, and c terms, and the MemorySheet object
    a_fn, b_fn, c_fn, mem_sheet = profile_mem_usage(
        OUTPUT_FUNCTION_DEGREE, OUTPUT_FUNCTION_DEGREE, OUTPUT_FUNCTION_DEGREE
    )
    print(f'Operation Successful')

    # This is functionally identical the C++ output function,
    # this is used to test the C++ output.
    # (Im currying this to add to the graph for each level)

    def f(level): return lambda degree: a_fn(level) * \
        degree ** 2 + b_fn(level) * degree + c_fn(level)

    # Print tested outputs for the resulting function
    if DEBUGGING:
        for level in range(1, 6):
            for degree in range(2, 7):
                debug(f'level={level} degree={degree}: {f(level-1)(degree-2)}')

    # Write C++ code for memory prediction to file
    write_mem(a_fn, b_fn, c_fn, get_commit_hash(PDE), get_commit_date(PDE))

    open(f'{CSV_OUTPUT_DIR}/{PDE}_mem.csv', 'w').write(str(mem_sheet))
    print(f'Wrote output memory spreadsheet to {CSV_OUTPUT_DIR}/{PDE}_mem.csv')

    # Show graphical output if requested by user
    graph(a_fn, b_fn, c_fn)


def graph(a_fn, b_fn, c_fn, f=None):
    if GRAPHING:
        # Graph the memory usage for each level function
        if f:
            p = Graph(title=PDE)
            for level in LEVELS:
                p.add_fn(f(level))
            p.plot()

        # Graph the accuracy of the a term function
        a = Graph(title='a term')
        a.add_fn(a_fn)
        a.plot()

        # Graph the accuracy of the b term function
        b = Graph(title='b term')
        b.add_fn(b_fn)
        b.plot()

        # Graph the accuracy of the c term function
        c = Graph(title='c term')
        c.add_fn(c_fn)
        c.plot()


def debug(*args):
    print("==[ DEBUG ]===>", *args)


# If this program is the target python script, run main
# if this script is imported by another, do not run main
if __name__ == '__main__':
    main()
