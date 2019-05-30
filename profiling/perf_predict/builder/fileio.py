from glob import glob
from options import MEM_HEADER_DIR, TIME_HEADER_DIR
from mem_predictor import MemPDE
from time_predictor import TimePDE


def get_mem_pdes():
    pdes = []
    for header in glob(f'{MEM_HEADER_DIR}/*.hpp'):
        pdes.append(
            MemPDE().from_header(header)
        )
    return pdes


def get_time_pdes():
    pdes = []
    for header in glob(f'{TIME_HEADER_DIR}/*.hpp'):
        pdes.append(
            TimePDE().from_header(header)
        )
    return pdes
