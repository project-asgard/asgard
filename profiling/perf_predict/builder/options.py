from argparse import ArgumentParser


parser = ArgumentParser(
    description='Packages output header files from the profiler into a single header to be used by ASGarD')

parser.add_argument('-m', '--mem-dir', dest='mem_dir',
                    required=True, help='The path to the folder containing the memory prediction header files for each of the PDEs')

parser.add_argument('-t', '--time-dir', dest='time_dir',
                    required=True, help='The path to the folder containing the time prediction header files for each of the PDEs')

parser.add_argument('-o', '--out-dir', dest='out_dir',
                    required=True, help='The output directory')

args = parser.parse_args()

# The dir where the input pde mem predictor headers are
MEM_HEADER_DIR = args.mem_dir

# The dir where the input pde time predictor headers are
TIME_HEADER_DIR = args.time_dir

# The dir for the generated output files
OUTPUT_DIR = args.out_dir

HEADER_NAME = 'predict.hpp'
CPP_NAME = 'predict.cpp'
