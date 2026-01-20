# this is a Doxygen directive used for documentation
## [bgk_py python]

import numpy as np
import os

import matplotlib.pyplot as plt

import asgard

if __name__ == '__main__':

    filename = "bgk_run.h5"
    asgard.run_with_args("./bgk", f"-of {filename}")

    snapshot = asgard.pde_snapshot(filename)

    if snapshot.num_dimensions == 2:
        # 1x1v case
        m0sh = snapshot.get_moment((0, ))
        m1sh = snapshot.get_moment((1, ))
        m2sh = snapshot.get_moment((2, ))

        m0, x = m0sh.plot_data1d(((), ), num_points = 64)
        m1, x = m1sh.plot_data1d(((), ), num_points = 64)
        m2, x = m2sh.plot_data1d(((), ), num_points = 64)

        plt.figure(1)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(x, m0, 'b', label = 'mass')
        u = m1 / m0
        ax1.plot(x, u, 'g', label = 'avg. speed')
        ax1.plot(x, m2 / m0 - u * u, 'r', label = 'temperature')

        ax1.legend()

        z, x, y = snapshot.plot_data2d(((), ()), num_points = 128)

        xmin = snapshot.dimension_min[0]
        ymin = snapshot.dimension_min[1]
        xmax = snapshot.dimension_max[0]
        ymax = snapshot.dimension_max[1]

        comp = ax2.imshow(np.flipud(z), cmap='turbo', extent=[xmin, xmax, ymin, ymax])

        plt.show()

## [bgk_py python]
