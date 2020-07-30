#!/usr/bin/env python3

# imports
import csv
import subprocess

# constants
BENCH_FILE = 'bench.txt'
RUN_ENTRIES = 2 # run structure: asgard args, average timestep
EMPTY_RUN = ""
ASGARD_PATH = "../../build/asgard" # is there a better way to do this? not relocatable...

# helper
def is_float(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

# execution
timings = dict()
with open(BENCH_FILE) as run_details:
	run_csv = csv.reader(run_details, delimiter=',')
	commit = next(run_csv)[0]
	print('bench commit: {}'.format(commit))
	for run in run_csv:
		assert(len(run) == RUN_ENTRIES), "run args and avg timestep required for all runs!"
		asgard_args, avg_timestep_ms = run[:RUN_ENTRIES]
		print(asgard_args)
		if is_float(avg_timestep_ms):
			timings[asgard_args] = avg_timestep_ms
		else:
			timings[asgard_args] = EMPTY_RUN

for args, timing in timings.items():
    print(args)
    run_cmd = [ASGARD_PATH] + args.split(' ')
    print(run_cmd)
    result = subprocess.run(run_cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')	
    print(result)

        
