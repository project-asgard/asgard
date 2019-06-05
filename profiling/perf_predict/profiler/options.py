from argparse import ArgumentParser


parser = ArgumentParser(description='Profiles memory usage for ASGarD')

parser.add_argument('-p', '--pde', dest='pde',
                    required=True, help='The PDE to profile')

parser.add_argument('-c', '--clean', dest='clean', action='store_const',
                    const=True, default=False, help='Don\'t use memory data collected by this program previously')
parser.add_argument('-d', '--debug', dest='debug', action='store_const',
                    const=True, default=False, help='Enable debugging output')
parser.add_argument('-g', '--graph', dest='graph', action='store_const',
                    const=True, default=False, help='Enable graph output')
parser.add_argument('-t', '--profile-time', dest='profile_time', action='store_const',
                    const=True, default=False, help='Profile time instead of memory')

parser.add_argument('-od', '--out-degree', dest='out_degree', default=4,
                    type=int, help='The degree of the functions that output the a, b, and c terms of the quadratic that describes memory')

parser.add_argument('-lmax', '--level-max', dest='level_max', default=5,
                    type=int, help='The maximum level value to profile')
parser.add_argument('-lmin', '--level-min', dest='level_min', default=1,
                    type=int, help='The minimum level value to profile')

parser.add_argument('-dmax', '--degree-max', dest='degree_max', default=6,
                    type=int, help='The maximum degree value to profile')
parser.add_argument('-dmin', '--degree-min', dest='degree_min', default=2,
                    type=int, help='The minimum degree value to profile')

parser.add_argument('-a', '--asgard-path', dest='asgard_path', default='../asgard',
                    help='The path to the asgard executable')

parser.add_argument('--data-dir', dest='data_dir', default='internal_data',
                    help='The output dir for the data files used internally by this program for generating prediction functions')

parser.add_argument('-o', '--output-dir', dest='output_dir', default='output',
                    help='The output directory for data readable by the user, such as spreadsheets and header files')

args = parser.parse_args()

# The name of the PDE
PDE = args.pde

# The range of levels to profile
LEVELS = list(range(args.level_min, args.level_max+1))
# The range of degrees to profile
DEGREES = list(range(args.degree_min, args.degree_max+1))

# Path to asgard executable
ASGARD_PATH = args.asgard_path

# Path to output data directory, where memory and time data is stored
DATA_FILE_DIR = args.data_dir + '/'

# The output dir for user readable data files such as spreadsheets and header files
OUTPUT_DIR = args.output_dir

# Name of the output header file containing C++ function for memory prediction
OUTPUT_MEM_HPP = f'{OUTPUT_DIR}/mem_pred/{PDE}.hpp'

# Name of the output header file containing C++ function for time prediction
OUTPUT_TIME_HPP = f'{OUTPUT_DIR}/time_pred/{PDE}.hpp'

# Has asgard already been profiled? Use old data to reuse previous profiles
USE_OLD_DATA = not args.clean

# Enable debugging output
DEBUGGING = args.debug

# Enable graph output
GRAPHING = args.graph

# Are we profiling time instead of memory??
PROFILING_TIME = args.profile_time

# The degree of the a, b, and c functions that generate the terms for the quadratic
OUTPUT_FUNCTION_DEGREE = args.out_degree

# The output dir for the csv spreadsheets
CSV_OUTPUT_DIR = f'{OUTPUT_DIR}/csv'
