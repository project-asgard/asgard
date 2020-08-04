#!/usr/bin/env python3

# imports
import csv
import subprocess
import re
import sys
import argparse
import os

# constants
RUN_ENTRIES = 2 # run structure: asgard args, average timestep
EMPTY_RUN = ''
ASGARD_PATH = './asgard' # is there a better way to do this? not relocatable...
NUM_THREADS = 8

## exit codes
UPDATE_BENCH_CODE=255
FAILURE_CODE = 1

## for regex
TIMESTEP_BEGIN = 'explicit_time_advance - avg: '
TIMESTEP_END = ' min:'
COMMIT_BEGIN = 'Commit Summary: '
COMMIT_END = '\n'
 
TOLERANCE = 8 # percent change in runtime we will accept

# helper
def is_float(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def find_first_between(haystack, begin, end):
	find = re.search(begin + '(.*)' + end, haystack)
	if find is None: 
		return find
	needle = find.group(1)
	return needle

def parse_timestep_avg(result):
    time = find_first_between(result, TIMESTEP_BEGIN, TIMESTEP_END)
    assert(is_float(time)), 'parsed avg timestep should be a float'
    print('time: ' + time)
    return(float(time))

def parse_commit(result):
	commit = find_first_between(result, COMMIT_BEGIN, COMMIT_END)
	assert(commit is not None), 'failed to parse commit hash'
	return commit

def within_performance_range(old_time, new_time):
	if old_time == new_time:
		diff = 0
	try:
		diff = (old_time - new_time) / old_time * 100.0
	except ZeroDivisionError:
		diff = float('inf')
	print('difference = {}'.format(diff))
	if abs(diff) > TOLERANCE:
		return False
	return True	

# execution
parser = argparse.ArgumentParser(description='--Run benchmark performance tests for ASGarD--')
parser.add_argument('bench_file', help='benchmark file with arguments and timings')
bench_file = parser.parse_args().bench_file

timings = dict()
with open(bench_file) as run_details:
	run_csv = csv.reader(run_details, delimiter=',')
	commit = next(run_csv)[0]
	print('bench commit: {}'.format(commit))
	for run in run_csv:
		assert(len(run) == RUN_ENTRIES), 'run args and avg timestep required for all runs'
		asgard_args, avg_timestep_ms = run[:RUN_ENTRIES]
		if is_float(avg_timestep_ms):
			timings[asgard_args] = float(avg_timestep_ms)
		else:
			timings[asgard_args] = EMPTY_RUN

empty_keys = [args for args, time in timings.items() if time == EMPTY_RUN]
assert(len(empty_keys) == 0 or len(empty_keys) == len(timings)), 'timings must be all present (check run) or none present (set run)' 
no_data = (len(empty_keys) == len(timings))

# run asgard to collect timing
new_times = dict()
os.environ['OMP_NUM_THREADS'] = '{}'.format(NUM_THREADS)
for args, timing in timings.items():
	run_cmd = [ASGARD_PATH] + args.split(' ')
	print('now running: {}'.format(args))
	result = subprocess.run(run_cmd, env=os.environ, 
						    stdout=subprocess.PIPE).stdout.decode('utf-8')	
	new_time = parse_timestep_avg(result)
	commit = parse_commit(result)
	if no_data:	# setting new benchmark
		new_times[args] = new_time
	else:
		if not within_performance_range(timing, new_time):
			print('test failed!')
			sys.exit(FAILURE_CODE)
print('run commit: {}'.format(commit))

# write benchmark if not present
if no_data:
	print('no benchmark set, writing new values now')
	with open(bench_file,'w') as bench:
		bench.write('{}\n'.format(commit))
		for args, time in new_times.items():
			bench.write('{},{}\n'.format(args, time)) 
	sys.exit(UPDATE_BENCH_CODE)

print('successful test!')
