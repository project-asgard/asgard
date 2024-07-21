
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
    if not os.path.isfile('example_continuity_2d'):
        print("You must first build this project using CMake, e.g.,")
        print("  mkdir build")
        print("  cd build")
        print("  cmake ..")
        print("  make -j")
        print("")
        print("Then run this script from the build folder")
        exit(1)

    print("asgard: running the continuity example")
    os.system("./example_continuity_2d -p continuity_2 -d 2 -l 6 -w 10 -n 10 -t 0.0001")

    # the example above will run for 10 time steps and the -w 10 options
    # will tell the code to output on the final 10-th step
    output_filename = 'asgard_wavelet_10.h5'
    if not os.path.isfile(output_filename):
        print("ERROR: example_continuity_2d did not generate an output file")
        exit(1)

    # using the ASGarD python module to read from the file
    snapshot = asgard.pde_snapshot(output_filename)

    print("problem has %d dimensions" % snapshot.num_dimensions)
    print("creating 2d plot")

    # the plot_data2d() method will generate inputs/outputs for 2d plotting
    # the outputed variables are:
    #  z is the value of the pde solution, it is a 2d array
    #  x is the first coordinate of the z points
    #  y is the second coordinate of the z points
    #  basically: z = u(t, x, y)
    # - when empty tuples are provided, the min/max values will be used
    # - we can zoom-in on a region with custom min/max values, e.g.,
    #    plot_data2d(((0, 1), (0, 2)), 128) will plot the first quadrant only
    # - for data with more then 2 dimensions, only two can be given ranges
    #   the rest must contain nominal values, e.g., for a 4d problem
    #    plot_data2d((0.01, (-1, 1), 0.3, ()), 128)
    #    meaning x0 = 0.01, x1 \in (-1, 1), x2 = 0.3, x3 \in (min, max)
    #  see also the 1d example below
    z, x, y = snapshot.plot_data2d(((), ()), num_points = 128)

    # the points are always taken stricktly in the interior of the domain
    # putting points right on the boundary may lead to instability due
    # to the discontinuous nature of the DG basis
    # we need to manually set the correct bounds on the plots
    xmin = snapshot.dimension_min[0]
    ymin = snapshot.dimension_min[1]
    xmax = snapshot.dimension_max[0]
    ymax = snapshot.dimension_max[1]

    # compare against the analytic solution
    h = np.cos(np.pi * x) * np.sin(2.0 * np.pi * y) * np.sin(2.0 * snapshot.time)

    # python plotting:
    # create a new figure with aspect ratio (14, 8)
    fig = plt.figure(1, figsize=(14, 8))
    # use side-by-side plots on a grid 1 by 2, using the plot at position (0, 0)
    ax = plt.subplot2grid((1, 2), (0, 0))
    # add the title
    ax.set_title("computed", fontsize = 18)
    # plot the image of the computed values
    comp = ax.imshow(z, cmap='jet', extent=[xmin, xmax, ymin, ymax])
    # set the colorbar
    fig.colorbar(comp, orientation='vertical')

    # plot on the cell at (0, 1) and plot the exact (analytic) solution
    ax = plt.subplot2grid((1, 2), (0, 1))
    ax.set_title("exact", fontsize = 18)
    exac = ax.imshow(h, cmap='jet', extent=[xmin, xmax, ymin, ymax])
    fig.colorbar(exac, orientation='vertical')

    print("creating 1d plot")

    z, y = snapshot.plot_data1d((0.01, ()), num_points = 128)

    h = np.cos(np.pi * 0.01) * np.sin(2.0 * np.pi * y) * np.sin(2.0 * snapshot.time)

    # create a new figure with aspect ratio (14, 8)
    # plot side-by-side as above but using line plots
    fig = plt.figure(2, figsize=(14, 7))

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax.set_title("computed", fontsize = 18)
    comp = ax.plot(y, z)
    ax = plt.subplot2grid((1, 2), (0, 1))
    ax.set_title("exact", fontsize = 18)
    comp = ax.plot(y, h)

    print("")  # prettier output

    plt.show()
