from glob import glob
from sys import argv
from os import system
from shutil import move
from os.path import dirname
from config import config
from options import ASGARD_PATH, PDEs, GRAPHING, OUTPUT_DIR, CLEANING


HOME = dirname(__file__)
if HOME == "":
    HOME = '.'
PROFILER = f'{HOME}/profiler'
BUILDER = f'{HOME}/builder'


for pde in PDEs:
    # Profile memory
    system(rf'python {PROFILER}/main.py -a {ASGARD_PATH} -p {pde} -lmax {config[pde]["mem"]["lmax"]} -dmax {config[pde]["mem"]["dmax"]} -od {config[pde]["time"]["od"]} {GRAPHING} --data-dir {HOME}/internal_data -o {HOME}/output {CLEANING}'
           )

    # Profile time
    system(
        rf'python {PROFILER}/main.py -a {ASGARD_PATH} -t -p {pde} -lmax {config[pde]["time"]["lmax"]} -dmax {config[pde]["time"]["dmax"]} -od {config[pde]["time"]["od"]} {GRAPHING} --data-dir {HOME}/internal_data -o {HOME}/output {CLEANING}')

system(
    rf'python {BUILDER}/main.py -m {HOME}/output/mem_pred -t {HOME}/output/time_pred -o {OUTPUT_DIR}')
