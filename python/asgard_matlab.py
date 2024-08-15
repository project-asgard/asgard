
import sys, os
import numpy as np
import scipy.io as sio

sys.path.append('@_asgard_matlab_pypath_@')
import asgard

def stat_file():
    filename = sys.argv[2]
    print('asgard stating: ', sys.argv[2])

    snapshot = asgard.pde_snapshot(filename)

    stats = {
        'filename' : filename,
        'num_dimensions' : snapshot.num_dimensions,
        'dimension_min'  : snapshot.dimension_min,
        'dimension_max'  : snapshot.dimension_max,
        'time'           : snapshot.time,
        'num_cells'      : snapshot.num_cells,
        }

    sio.savemat('__asgard_pymatlab.mat', stats)

def read_points():
    data = sio.loadmat('__asgard_pymatlab.mat')
    pnts = data['point_list']
    nump = int(data['num_points'][0])

    llist = []

    ndim = pnts.shape[1]
    for i in range(ndim):
        if pnts[0][i].shape[1] == 1:  # single entry
            llist.append(pnts[0][i][0][0])
        elif pnts[0][i].shape[1] == 0:  # null entry
            llist.append([])
        elif pnts[0][i].shape[2] == 0:  # range entry
            llist.append([pnts[0][i][0][0], pnts[0][i][0][1]])
        else:
            raise TypeError("cannot understand the matlab file, shape[1] for an entry is not 0, 1, or 2")

    return llist, nump

def plot1d():
    filename = sys.argv[2]
    print('plotting 1d: ', sys.argv[2])

    snapshot = asgard.pde_snapshot(filename)

    llist, nump = read_points()
    print(llist)
    print(nump)

    print(len(llist), snapshot.num_dimensions)

    z, x = snapshot.plot_data1d(llist, nump)

    sio.savemat('__asgard_pymatlab.mat', {'z' : z, 'x' : x})

def plot2d():
    filename = sys.argv[2]
    print('plotting 2d: ', sys.argv[2])

    snapshot = asgard.pde_snapshot(filename)

    llist, nump = read_points()

    z, x, y = snapshot.plot_data2d(llist, nump)

    sio.savemat('__asgard_pymatlab.mat', {'z' : z, 'x' : x, 'y' : y})

def evaluate():
    filename = sys.argv[2]
    print('evaluating: ', sys.argv[2])

    snapshot = asgard.pde_snapshot(filename)

    data = sio.loadmat('__asgard_pymatlab.mat')

    z = snapshot.evaluate(data['points'])

    sio.savemat('__asgard_pymatlab.mat', {'z' : z})

def cell_centers():
    filename = sys.argv[2]
    print('sparse grid from: ', sys.argv[2])

    snapshot = asgard.pde_snapshot(filename)

    z = snapshot.cell_centers()

    sio.savemat('__asgard_pymatlab.mat', {'z' : z})

if __name__ == '__main__':
    if sys.argv[1] == '-stat':
        stat_file()
    elif sys.argv[1] == '-plot1d':
        plot1d()
    elif sys.argv[1] == '-plot2d':
        plot2d()
    elif sys.argv[1] == '-eval':
        evaluate()
    elif sys.argv[1] == '-getcells':
        cell_centers()
    else:
        raise Exception("unknown switch (-stat, -plot1d, -plot2d, -eval)")
