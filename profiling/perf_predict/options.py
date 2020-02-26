from argparse import ArgumentParser

parser = ArgumentParser(
    description='Profile ASGarD and build header file containing functions to predict memory usage and compute time')


parser.add_argument('asgard_path', metavar='ASGarD Path', type=str,
                    help='The path to the asgard executable')
parser.add_argument('pdes', metavar='PDEs', type=str, nargs='+',
                    help='The PDEs to profile')
parser.add_argument('-g', '--graph', dest='graph', action='store_const',
                    const=True, default=False, help='Enable graph output')
parser.add_argument('-d', '--debug', dest='debug', action='store_const',
                    const=True, default=False, help='Enable debugging')
parser.add_argument('-o', '--output-dir', dest='output_dir',
                    default='../src', help='Output directory of predict.hpp')

args = parser.parse_args()


ASGARD_PATH = args.asgard_path
PDEs = args.pdes
GRAPHING = '-g' if args.graph else ''
CLEANING = '-c' if not args.debug else ''
OUTPUT_DIR = args.output_dir
