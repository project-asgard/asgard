from functools import reduce
from subprocess import check_call, STDOUT
from sys import argv
from os.path import isfile
from options import DATA_FILE_DIR, ASGARD_PATH, USE_OLD_DATA


# Runs `time` on the commandline to get time
def run_time(level, degree, pde, time_output):
    with open(f'{time_output}', 'w') as output_file:
        check_call([
            'time',
            f'{ASGARD_PATH}',
            '-l', f'{level}',
            '-d', f'{degree}',
            '-p', f'{pde}'
        ], stdout=output_file, stderr=STDOUT)
        output_file.close()


# Gets the actual compute time in seconds from asgard
def get_time(level, degree, pde):
    # The extension added to the time output file
    time_output = f"{DATA_FILE_DIR}/time.out.l{level}_d{degree}_p{pde}"

    # Call time on asgard with specified level, degree, and pde
    # Write output to time_output
    if not USE_OLD_DATA or not isfile(time_output):
        run_time(level, degree, pde, time_output)

    # Parse the output file
    seconds = TimeReader(time_output).get_seconds()

    return seconds


# Used to read the time commandline output from running asgard
class TimeReader:
    def __init__(self, filename=None):
        self.filename = filename
        if filename:
            self.open(filename)

    # Open a time output file
    def open(self, filename):
        self.filename = filename
        self.lines = open(filename, 'r').readlines()

    # Get the seconds used by asgard
    def get_seconds(self):
        # 0.21user 0.40system 0:00.11elapsed 537%CPU (0avgtext+0avgdata 7088maxresident)k
        for line in self.lines:
            if "elapsed" in line:
                time_str = line.split('elapsed')[0].split('system')[-1]
                minutes = float(time_str.split(':')[0])
                seconds = float(time_str.split(':')[1])
                seconds += minutes * 60
                return seconds

        raise Exception(f"No time found in time file '{self.filename}'")
