#!/bin/env python

import os
import sys

import h5py

import matplotlib.pyplot as plt

def read_data(filename, dataset):
    data_file = h5py.File(filename, 'r')

    print(data_file)
    print(data_file.keys())

    nodes = data_file['nodes'][()]
    soln = data_file['soln'][()]

    tmp = data_file['soln'][()]

    fig, ax = plt.subplots()
    ax.contourf(nodes, nodes, tmp.reshape((len(nodes), len(nodes))).transpose())
    ax.set_title("t = {}".format(data_file['time'][()]))
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise RuntimeError("Expected a datafile")

    input_fname = sys.argv[1]

    if not os.path.exists(input_fname):
        raise RuntimeError("File '{}' does not exist".format(input_fname))

    read_data(input_fname, 'asgard')
