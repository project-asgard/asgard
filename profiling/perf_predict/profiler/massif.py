from functools import reduce
from subprocess import check_call, STDOUT, DEVNULL
from sys import argv
from os.path import isfile
from options import DATA_FILE_DIR, ASGARD_PATH, USE_OLD_DATA, LEVELS, DEGREES


# Convert bytes to megabytes
def bytes_to_megabytes(b): return int(b) / 10**6


# Runs massif on the commandline to generate output files for profiling
def run_massif(level, degree, pde, massif_output, asgard_output):
    with open(f'{asgard_output}', 'w') as output_file:
        check_call([
            'valgrind',
            '--tool=massif',
            f'--massif-out-file={massif_output}',
            f'{ASGARD_PATH}',
            '-l', f'{level}',
            '-d', f'{degree}',
            '-p', f'{pde}'
        ], stdout=output_file, stderr=STDOUT)
        output_file.close()


# Gets the workspace mem usage and total mem usage for asgard
def get_mem_usage(level, degree, pde):
    # The extension added to the massif and asgard output files
    output_file_extention = f".out.l{level}_d{degree}_p{pde}"

    massif_output = DATA_FILE_DIR + "/massif" + output_file_extention
    asgard_output = DATA_FILE_DIR + "/asgard" + output_file_extention

    # Call valgrind on asgard with specified level, degree, and pde
    # Write output to asgard_output
    if not USE_OLD_DATA or not isfile(massif_output):
        run_massif(level, degree, pde, massif_output, asgard_output)

    # Parse the output files
    total_mem_usage = MassifReader(massif_output).get_peak()

    # A struct containing the workspace and total mem usage for asgard
    return total_mem_usage

# Used to read the massif output file from running asgard


class MassifReader:
    def __init__(self, filename=None):
        self.filename = filename
        if filename:
            self.open(filename)

    # Open a massif output file
    def open(self, filename):
        self.filename = filename
        self.lines = open(filename, 'r').readlines()

    # Get the peak memory usage from the opened massif output file
    def get_peak(self):
        for i, line in enumerate(self.lines):
            if "heap_tree=peak" in line:
                heap_mb = bytes_to_megabytes(self.lines[i-3].split('=')[1])
                heap_extra_mb = bytes_to_megabytes(
                    self.lines[i-2].split('=')[1])
                return heap_mb + heap_extra_mb
        raise Exception(
            f"No peak found in massif output file '{self.filename}'")
