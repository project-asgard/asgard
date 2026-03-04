# this is a Doxygen directive used for documentation
## [bgk_py python]

import numpy as np
import os

import matplotlib.pyplot as plt

import asgard

if __name__ == '__main__':

    # most of the code in this file is related to Python matplotlib
    # the most notable ASGarD methods are
    #     run_with_args()
    #     pde_snapshot()
    #     get_aux_field()
    #     get_moment()
    #     plot_data2d()

    filename = "bgk_run.h5"
    # the "run_with_args" method will the provided executable file with the provided
    # string with arguments and it will also append all arguments used in the call
    # to this python script, e.g.,
    #     python3 -m bgk.py -m 8 -a 1.E-6 -n 20
    # will result in a call
    #     ./bgk -of bgk_run.h5 -m 8 -a 1.E-6 -n 20
    #
    # run_with_args() accepts an additional list of arguments, e.g., modified list of sys.argv
    # for example:
    #     python3 -m bgk.py run -m 8
    # then modify the code:
    #     mylist = sys.argv
    #     mylist.remove("run") if "run" in mylist else None
    #     asgard.run_with_args("./bgk", f"-of {filename}", mylist) # runs ./bgk -of bgk_run.h5 -m 8
    #
    asgard.run_with_args("./bgk", f"-of {filename}")

    snapshot = asgard.pde_snapshot(filename)

    # get the boundaries for the domain, this is needed to set the correct size
    # for the matplotlib imshow command
    xmin = snapshot.dimension_min[0]
    ymin = snapshot.dimension_min[1]
    xmax = snapshot.dimension_max[0]
    ymax = snapshot.dimension_max[1]

    if "poisson" in snapshot.subtitle:
        # 1x1v case, poisson
        # plotting the initial and final perturbations, i.e., aux fields

        # Try running this with the following:
        #   python3 bgk.py -poisson -m 9 -a 1.E-6 -t 2

        assert snapshot.num_position == 1 and snapshot.num_velocity == 1

        # the aux fields can be requested either by name or by index,
        # e.g., init_pert  = snapshot.get_aux_field(0)
        init_pert  = snapshot.get_aux_field("initial perturbation")
        final_pert = snapshot.get_aux_field("final perturbation")

        # generating two 2D plots
        z0, x, y = init_pert.plot_data2d(((), ()), num_points = 128)
        z1, x, y = final_pert.plot_data2d(((), ()), num_points = 128)

        fig, (ax0, ax1) = plt.subplots(1, 2)

        fig.suptitle(snapshot.title)

        ax0.set_title(init_pert.title + ", t = 0")
        img0 = ax0.imshow(np.flipud(z0), cmap='turbo', extent=[xmin, xmax, ymin, ymax])
        ax0.set_xlabel("x", fontsize = 'large')
        ax0.set_ylabel("v", fontsize = 'large')
        fig.colorbar(img0, ax=ax0, orientation='vertical')

        ax1.set_title(final_pert.title + f", t = {final_pert.time:.4f}")
        img1 = ax1.imshow(np.flipud(z1), cmap='turbo', extent=[xmin, xmax, ymin, ymax])
        ax1.set_xlabel("x", fontsize = 'large')
        ax1.set_ylabel("v", fontsize = 'large')
        fig.colorbar(img1, ax=ax1, orientation='vertical')

        plt.show()

    elif snapshot.num_dimensions == 2:
        # 1x1v case, shock1d

        # When using high collision frequency, the density develops a stair-case pattern.
        #
        # Try running this with:
        #   python3 bgk.py -shock1d -nu 1000 -m 8 -a 1.E-5 -n 2000

        assert snapshot.num_position == 1 and snapshot.num_velocity == 1

        # obtaining the auxiliary fields associated with the moments
        # the 3 variables are another instances of the snapshot class
        m0sh = snapshot.get_moment((0, ))
        m1sh = snapshot.get_moment((1, ))
        m2sh = snapshot.get_moment((2, ))

        m0, x = m0sh.plot_data1d(((), ), num_points = 128)
        m1, x = m1sh.plot_data1d(((), ), num_points = 128)
        m2, x = m2sh.plot_data1d(((), ), num_points = 128)

        fig, (ax0, ax1) = plt.subplots(1, 2)

        fig.suptitle(snapshot.title + f" ({snapshot.subtitle})")

        # instead of plotting the "raw" moment, we compute the fluid variables
        # density, velocity and temperature
        ax0.set_title("fluid variables")
        ax0.plot(x, m0, 'b', label = 'density')
        u = m1 / m0
        ax0.plot(x, u, 'g:', label = 'avg. velocity')
        ax0.plot(x, m2 / m0 - u * u, 'r-.', label = 'temperature')
        ax0.set_xlabel("x", fontsize = 'large')
        ax0.set_ylabel("value", fontsize = 'large')

        ax0.legend()

        z, x, y = snapshot.plot_data2d(((), ()), num_points = 128)

        ax1.set_title(f"solution at t = {snapshot.time:.4f}")
        img1 = ax1.imshow(np.flipud(z), cmap='viridis', extent=[xmin, xmax, ymin, ymax])
        ax1.set_xlabel("x", fontsize = 'large')
        ax1.set_ylabel("v", fontsize = 'large')
        fig.colorbar(img1, ax=ax1, orientation='vertical')

        plt.show()

    elif snapshot.num_dimensions == 4:
        # 2x2v case, shock2d

        # This is NOT the proper way to manage the 2D BGK example.
        # Even on a good machine, running the 2D problem can take in order of hours,
        # thus, it is better to separate the running and plotting logic.
        # Such split is beyond the scope of this example,
        # but look at the comment related to run_with_args() and custom arguments.

        # example command:
        #   python3 bgk.py -shock2d -nu 100 -m 8 -a 1.E-5 -n 2000
        #
        # This is nice visual example but also takes a while to compute.
        # It requires up to 90 million degrees of freedom and minimum 16GB GPU
        # or 32GB of system RAM. On a workstation Nvidia GPU this takes little over 2 hours.

        assert snapshot.num_position == 2 and snapshot.num_velocity == 2

        # the moments are now defined by 2D tuples
        m0sh = snapshot.get_moment((0, 0))
        m10sh = snapshot.get_moment((1, 0))
        m01sh = snapshot.get_moment((0, 1))
        m20sh = snapshot.get_moment((2, 0))
        m02sh = snapshot.get_moment((0, 2))

        m0, x, y = m0sh.plot_data2d(((), ()), num_points = 128)
        m10, x, y = m10sh.plot_data2d(((), ()), num_points = 128)
        m01, x, y = m01sh.plot_data2d(((), ()), num_points = 128)
        m20, x, y = m20sh.plot_data2d(((), ()), num_points = 128)
        m02, x, y = m02sh.plot_data2d(((), ()), num_points = 128)


        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

        fig.suptitle(snapshot.title + f" ({snapshot.subtitle})")

        # instead of plotting the "raw" moment, we compute the fluid variables
        # density, velocity and temperature
        ax0.set_title("density")
        img0 = ax0.imshow(np.flipud(m0), cmap='viridis', extent=[xmin, xmax, ymin, ymax])
        fig.colorbar(img0, ax=ax0, orientation='vertical')
        ax0.set_xlabel("x1", fontsize = 'large')
        ax0.set_ylabel("x2", fontsize = 'large')

        ax1.set_title("avg. speed")
        u0 = m10 / m0
        u1 = m01 / m0
        aspd = np.sqrt(u0 * u0 + u1 * u1)
        img1 = ax1.imshow(np.flipud(aspd), cmap='turbo', extent=[xmin, xmax, ymin, ymax])
        fig.colorbar(img1, ax=ax1, orientation='vertical')
        ax1.set_xlabel("x1", fontsize = 'large')
        ax1.set_ylabel("x2", fontsize = 'large')

        ax2.set_title("temperature")
        temp = 0.5 * (m20 + m02) / m0 - u0 * u0 - u1 * u1
        img2 = ax2.imshow(np.flipud(temp), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        fig.colorbar(img2, ax=ax2, orientation='vertical')
        ax2.set_xlabel("x1", fontsize = 'large')
        ax2.set_ylabel("x2", fontsize = 'large')

        plt.show()

## [bgk_py python]
