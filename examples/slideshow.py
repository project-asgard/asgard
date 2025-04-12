# this is a Doxygen directive used for documentation
## [slideshow python]

import numpy as np
import os, sys

import matplotlib.pyplot as plt

import asgard

# this example runs a slideshow of the aux fields stored in an asgard file

if __name__ == '__main__':
    if len(sys.argv) < 2 or "help" in sys.argv[1] or not os.path.isfile(sys.argv[1]):
        print(f"\nusage: python3 {sys.argv[0]} <filename>")
        print("  the <filename> should be an asgard hdf5 file with stored aux fields\n")
        exit(1)

    maindata = asgard.pde_snapshot(sys.argv[1])

    if len(maindata.aux_fields) == 0:
        print(f" the file {sys.argv[1]} does not contain aux-fields")
        exit(1)

    # plot dimensions 0 and 1, pad with 0
    view = [(), ()]
    while len(view) < maindata.num_dimensions:
        view.append(0.0)

    # the min/max of the dimensions
    xmin = maindata.dimension_min[0]
    ymin = maindata.dimension_min[1]
    xmax = maindata.dimension_max[0]
    ymax = maindata.dimension_max[1]

    for i in range(len(maindata.aux_fields)):
        islast = (i == len(maindata.aux_fields) - 1)

        shot = maindata.get_aux_field(i)

        z, x, y = shot.plot_data2d(view, num_points = 256)

        plt.figure(1)
        plt.clf()

        plt.title(shot.title, fontsize = 'large')

        p = plt.imshow(np.flipud(z), cmap='jet', extent=[xmin, xmax, ymin, ymax])

        plt.colorbar(p, orientation='vertical')

        plt.gca().set_anchor('C')

        plt.xlabel(shot.dimension_names[0], fontsize='large')
        plt.ylabel(shot.dimension_names[1], fontsize='large')

        plt.show(block=islast)

        if not islast:
            plt.pause(1)

## [slideshow python]
