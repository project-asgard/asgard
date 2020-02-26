from tqdm import tqdm
from spreadsheet import TimeSheet, MemorySheet
from regression import Function
from massif import get_mem_usage
from timer import get_time
from options import LEVELS, DEGREES, PDE


# Generates three functions a, b, and c with the specified degree.
# These functions describe the a, b, and c terms of the quadratic function
# that describes the compute time for the PDE.
def profile_time(a_degree=4, b_degree=4, c_degree=4):
    print(f'Profiling ASGarD\'s compute time for {PDE}...')

    # The output sheet describing time data
    time_sheet = TimeSheet()

    # The function describing the time used for each level for every degree
    f = Function(degree=2)  # a quadratic function

    # a list containing the terms of the function describing time used as a function of degree per level
    level_terms = []

    # Iterate over level and degrees for the PDE
    for level in tqdm(LEVELS, desc=f'Total progress'):
        for i, degree in enumerate(tqdm(DEGREES, leave=False, desc=f'Level {level}')):
            seconds = []

            for _ in range(3):
                seconds.append(get_time(level, degree, PDE))

            average_seconds = sum(seconds) / len(seconds)

            # time as a function of degree per level
            f.add_point(i, average_seconds)
            time_sheet.add_seconds(level, degree, average_seconds)

        # add to the list of the list of terms per level's time function
        level_terms.append(f.as_terms())

        # clear the current f to reuse for the next level's time function
        f.clear()

    # Functions describing the a, b, and c terms of the function describing seconds used
    a = Function(degree=a_degree)
    b = Function(degree=b_degree)
    c = Function(degree=c_degree)

    # Add the points for the a, b, and c functions
    for i, terms in enumerate(level_terms):
        a.add_point(i, terms[0])
        b.add_point(i, terms[1])
        c.add_point(i, terms[2])

    return a, b, c, time_sheet


# Generates three functions a, b, and c with the specified degree.
# These functions describe the a, b, and c terms of the quadratic function
# that describes the memory usage of the PDE.
def profile_mem_usage(a_degree=4, b_degree=4, c_degree=4):
    print(f'Profiling ASGarD\'s memory usage for {PDE}...')

    # The output sheet describing memory data
    mem_sheet = MemorySheet()

    # The function describing the memory usage for each level for every degree
    f = Function(degree=2)  # a quadratic function

    # a list containing the terms of the function describing memory usage as a function of degree per level
    level_terms = []

    # Iterate over level and degrees for the PDE
    for level in tqdm(LEVELS, desc=f'Total progress'):
        for i, degree in enumerate(tqdm(DEGREES, leave=False, desc=f'Level {level}')):
            mem_usage = get_mem_usage(level, degree, PDE)

            # total memory usage as a function of degree
            f.add_point(i, mem_usage)
            mem_sheet.add_data(
                level, degree, mem_usage
            )

        # add to the list of the list of terms per level memory function
        level_terms.append(f.as_terms())

        # clear the current f to reuse
        f.clear()

    # Functions describing the a, b, and c terms of the function describing memory usage
    a = Function(degree=a_degree)
    b = Function(degree=b_degree)
    c = Function(degree=c_degree)

    # Add the points for the a, b, and c functions
    for i, terms in enumerate(level_terms):
        a.add_point(i, terms[0])
        b.add_point(i, terms[1])
        c.add_point(i, terms[2])

    return a, b, c, mem_sheet
