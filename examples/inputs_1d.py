
import numpy as np
import os

import matplotlib.pyplot as plt

import asgard

# This exmaple is related to continuity_2d.cpp
# 1. This will run the C++ executable and generate an hdf5 file
#    - The file contains a snapshot of the PDE solution
# 2. The snapshot is loaded using the asgard python bindings
# 3. The solution and the exact solution are plotted together

if __name__ == '__main__':
    # build folder and install folder names can be different
    if os.path.isfile('inputs_1d'):
        exefilename = 'inputs_1d'
    elif os.path.isfile('example_inputs_1d'):
        exefilename = 'example_inputs_1d'
    else:
        print("You must first build this project using CMake, e.g.,")
        print("  mkdir build")
        print("  cd build")
        print("  cmake ..")
        print("  make -j")
        print("")
        print("Then run this script from the build folder")
        exit(1)

    print("running the inputs examples")

    outfile1 = 'waves1.h5'
    outfile2 = 'waves2.h5'

    # providing the input file and also forcing the degree to 1
    os.system("./{} -if inputs_1d_1.txt -d 2 -of {}".format(exefilename, outfile1))
    os.system("./{} -if inputs_1d_2.txt -d 2 -of {}".format(exefilename, outfile2))

    if not os.path.isfile(outfile1) or not os.path.isfile(outfile2):
        print("ERROR: example_inputs_1d did not generate an output file")
        exit(1)

    # using the ASGarD python module to read from the file
    snapshot1 = asgard.pde_snapshot(outfile1)
    snapshot2 = asgard.pde_snapshot(outfile2)

    print(" -- using inputs_1d_1.txt --")
    print("problem title: %s" % snapshot1.title)
    print("problem   sub: %s" % snapshot1.subtitle)
    print("num cells:     %s" % snapshot1.num_cells)
    print("final time:    %s" % snapshot1.time)
    print(" -- using inputs_1d_2.txt --")
    print("problem title: %s" % snapshot2.title)
    print("problem   sub: %s" % snapshot2.subtitle)
    print("num cells:     %s" % snapshot2.num_cells)
    print("final time:    %s" % snapshot2.time)
    print("")

    print("creating plots")

    # see the continuity_2d example for details on the plotting calls

    z1, x1 = snapshot1.plot_data1d(((), ), num_points = 1024)
    z2, x2 = snapshot2.plot_data1d(((), ), num_points = 1024)

    # expected exact solution
    h1 = np.sin(x1) * np.cos(snapshot1.time)
    h2 = np.sin(x2) * np.cos(snapshot1.time)

    # create a new figure with aspect ratio (14, 8)
    # plot side-by-side as above but using line plots
    fig = plt.figure(1, figsize=(14, 7))

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax.set_title("computed", fontsize = 18)
    comp = ax.plot(x1, z1)
    ax = plt.subplot2grid((1, 2), (0, 1))
    ax.set_title("exact", fontsize = 18)
    comp = ax.plot(x1, h1)

    fig = plt.figure(2, figsize=(14, 7))

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax.set_title("computed", fontsize = 18)
    comp = ax.plot(x2, z2)
    ax = plt.subplot2grid((1, 2), (0, 1))
    ax.set_title("exact", fontsize = 18)
    comp = ax.plot(x2, h2)

    print("")  # prettier output

    plt.show()
