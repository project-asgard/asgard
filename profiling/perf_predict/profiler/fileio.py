'''
This module encapsulates (almost) all file interactions.

The purpose of this is so you never have to dig deep
into the rest of the program to change how a file is
read or written to.

This is mainly focused on the parts of the program that output
data used by another program, being consistent about file interactions
across programs is very important for the stability of this tool.
'''
from os import makedirs
from os.path import isdir, dirname
from options import OUTPUT_MEM_HPP, OUTPUT_TIME_HPP, DATA_FILE_DIR, CSV_OUTPUT_DIR, PDE
from datetime import datetime as dt

MEM_PDE_FN_NAME = f'{PDE}_MB'
TIME_PDE_FN_NAME = f'{PDE}_seconds'


# Check if memory header file output folder exists
if not isdir(dirname(OUTPUT_MEM_HPP)):
    makedirs(dirname(OUTPUT_MEM_HPP))

# Check if time header file output folder exists
if not isdir(dirname(OUTPUT_TIME_HPP)):
    makedirs(dirname(OUTPUT_TIME_HPP))

# Check if csv output folder exists
if not isdir(CSV_OUTPUT_DIR):
    makedirs(CSV_OUTPUT_DIR)

# Check if memory profiling data folder exists
if not isdir(DATA_FILE_DIR):
    makedirs(DATA_FILE_DIR)


# Write C++ code for memory to file
def write_mem(a_fn, b_fn, c_fn, commit_hash="", commit_date=""):
    with open(OUTPUT_MEM_HPP, 'w') as output_mem_hpp:
        output_mem_hpp.write(rf'''std::pair<std::string, double> {MEM_PDE_FN_NAME}(int level, int degree)
{{
    level -= 1;
    degree -= 2;

    double a = {a_fn.as_cpp(var='level')};
    double b = {b_fn.as_cpp(var='level')};
    double c = {c_fn.as_cpp(var='level')};

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        {'""' if commit_hash == '' else f'"(Predicted for {commit_hash} on {commit_date})"'},
        a * pow(degree, 2) + b * degree + c
    );
}}''')

    print(f'Wrote output C++ code for memory prediction to {OUTPUT_MEM_HPP}')


# Write C++ code for time to file
def write_time(a_fn, b_fn, c_fn, commit_hash="", commit_date=""):
    with open(OUTPUT_TIME_HPP, 'w') as output_mem_hpp:
        output_mem_hpp.write(rf'''std::pair<std::string, double> {TIME_PDE_FN_NAME}(int level, int degree)
{{
    level -= 1;
    degree -= 2;

    double a = {a_fn.as_cpp(var='level')};
    double b = {b_fn.as_cpp(var='level')};
    double c = {c_fn.as_cpp(var='level')};

    // return a * pow(degree, 2) + b * degree + c;

    return std::make_pair(
        {'""' if commit_hash == '' else f'"(Predicted for {commit_hash} on {commit_date})"'},
        a * pow(degree, 2) + b * degree + c
    );
}}''')

    print(f'Wrote output C++ code for time prediction to {OUTPUT_TIME_HPP}')
