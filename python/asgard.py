import sys

from ctypes import c_char_p, c_int, c_int64, c_double, c_float, c_void_p, POINTER, CDLL, create_string_buffer, RTLD_GLOBAL
import numpy as np

import h5py # required for now, maybe add lighter module later

_matplotlib_found_ = True
try:
    import matplotlib.pyplot as asgplot
except:
    _matplotlib_found_ = False

from asgard_config import __version__, __author__, __pyasgard_libasgard_path__

libasgard = CDLL(__pyasgard_libasgard_path__, mode = RTLD_GLOBAL)

libasgard.asgard_make_dreconstruct_solution.restype = c_void_p
libasgard.asgard_make_freconstruct_solution.restype = c_void_p

libasgard.asgard_make_dreconstruct_solution.argtypes = [c_int, c_int64, POINTER(c_int), c_int, POINTER(c_double)]
libasgard.asgard_make_freconstruct_solution.argtypes = [c_int, c_int64, POINTER(c_int), c_int, POINTER(c_float)]

libasgard.asgard_pydelete_reconstruct_solution.argtypes = [c_void_p, ]

libasgard.asgard_print_version_help.argtypes = []

libasgard.asgard_reconstruct_solution_setbounds.argtypes = [c_void_p, POINTER(c_double), POINTER(c_double)]
libasgard.asgard_reconstruct_solution.argtypes = [c_void_p, POINTER(c_double), c_int, POINTER(c_double)]
libasgard.asgard_reconstruct_cell_centers.argtypes = [c_void_p, POINTER(c_double)]

class pde_snapshot:
    '''
    Reads an ASGarD HDF5 file with wavelet information and reconstrcts data
    for plotting.

    Example:
      import asgard
      import matplotlib.pyplot as plt

      snapshot = asgard.pde_snapshot("asgard_wavelet_10.h5")
      z, x, y = snapshot.plot_data2d([[-1.0, 1.0], [-2.0, 2.0]], nump = 100)
      plt.imshow(h, cmap='jet', extent=[-1.0, 1.0, -2.0, 2.0])
    '''
    def __del__(self):
        if self.recsol != None:
            libasgard.asgard_pydelete_reconstruct_solution(self.recsol)

    def __init__(self, filename, verbose = False):
        self.verbose = verbose
        if self.verbose:
            print(' -- reading from: %s' % filename)

        with h5py.File(filename, "r") as fdata:
            # keep this for reference of the keys that we may need
            # print(fdata.keys())

            self.title    = fdata['title'][()].decode("utf-8")
            self.subtitle = fdata['subtitle'][()].decode("utf-8")

            self.num_dimensions = fdata['ndims'][()]
            self.degree         = fdata['degree'][()]

            self.state = fdata['state'][()]
            self.cells = fdata['elements'][()]
            self.time  = fdata['time'][()] # numeric time

            self.timer_report = fdata['timer_report'][()].decode("utf-8")

            self.num_cells = int(len(self.cells) / (2 * self.num_dimensions))

            self.dimension_names = [None for i in range(self.num_dimensions)]
            self.dimension_min = np.zeros((self.num_dimensions,))
            self.dimension_max = np.zeros((self.num_dimensions,))
            for i in range(self.num_dimensions):
                self.dimension_names[i] = fdata['dim{0}_name'.format(i)][()].decode("utf-8")
                self.dimension_min[i] = fdata['dim{0}_min'.format(i)][()]
                self.dimension_max[i] = fdata['dim{0}_max'.format(i)][()]

        # for plotting purposes, say aways from the domain edges
        # rounding error at the edge may skew the plots
        self.eps = 1.E-6 * np.min(self.dimension_max - self.dimension_min)

        if self.verbose:
            if self.state.dtype == np.float64:
                print('using double precision data (double) - 64-bit')
            else:
                print('using single precision data (float) - 32-bit')
            print('dimensions:  {0}'.format(self.num_dimensions))

            print('domain ranges')
            for i in range(self.num_dimensions):
                print('  {0}:  {1: <16f}  {2: <16f}'
                        .format(i, self.dimension_min[i], self.dimension_max[i]))

            print('number of sparse grid cells: %d' % self.num_cells)

        self.recsol = None
        if self.state.dtype == np.float64:
            self.double_precision = True
            self.recsol = libasgard.asgard_make_dreconstruct_solution(
                self.num_dimensions, self.num_cells, np.ctypeslib.as_ctypes(self.cells.reshape(-1,)),
                self.degree, np.ctypeslib.as_ctypes(self.state.reshape(-1,)))
        else:
            self.double_precision = False
            self.recsol = libasgard.asgard_make_freconstruct_solution(
                self.num_dimensions, self.num_cells, np.ctypeslib.as_ctypes(self.cells.reshape(-1,)),
                self.degree, np.ctypeslib.as_ctypes(self.state.reshape(-1,)))

        libasgard.asgard_reconstruct_solution_setbounds(self.recsol,
                                                        np.ctypeslib.as_ctypes(self.dimension_min.reshape(-1,)),
                                                        np.ctypeslib.as_ctypes(self.dimension_max.reshape(-1,)))

    def plot_data1d(self, dims, num_points = 32):
        '''
        Generates two 1d arrays (vals, x), so that vals is the values
        of the computed solution at the corresponding points in x.
        The array x is constructed from numpy linspace().
        The min/max values of x are provided in the dims list.
        The dims list also holds nominal values for the fixed dimensions.
        '''
        assert len(dims) == self.num_dimensions, "the length of dims (%d) must match num_dimensions (%d)" % (len(dims), self.num_dimensions)

        irange = -1
        minmax = None
        for i in range(self.num_dimensions):
            if isinstance(dims[i], (list, tuple)):
                assert len(dims[i]) == 2 or len(dims[i]) == 0, "elements of dims must be scalars or tuples/lists of len 2 or 0"
                assert irange == -1, "there should be only one dimension with a range in dims"
                irange = i
                if len(dims[i]) == 2:
                    assert self.dimension_min[i] <= dims[i][0] and dims[i][0] <= self.dimension_max[i], "range min for dimension %d is out of bounds" %i
                    assert self.dimension_min[i] <= dims[i][1] and dims[i][1] <= self.dimension_max[i], "range max for dimension %d is out of bounds" %i
                    minmax = (dims[i][0], dims[i][1])
                else:
                    minmax = (self.dimension_min[i], self.dimension_max[i])
            else:
                assert self.dimension_min[i] <= dims[i] and dims[i] <= self.dimension_max[i], "nominal value for dimension %d is out of bounds" %i

        assert irange > -1, "there should be one dimension with a range in dims"

        x = np.linspace(minmax[0] + self.eps, minmax[1] - self.eps, num_points)

        pnts = np.zeros((x.size, self.num_dimensions))
        for i in range(self.num_dimensions):
            if i == irange:
                pnts[:, i] = x
            else:
                pnts[:, i] = dims[i] * np.ones(x.shape)

        return self.evaluate(pnts).reshape(x.shape), x

    def plot_data2d(self, dims, num_points = 32):
        '''
        Generates three 2d arrays (vals, x1, x2), so that vals is the values
        of the computed solution at the corresponding points in x1 and x2.
        The x1/2 arrays are constructed from numpy linspace() and meshgrid().
        The min/max values of x1/2 are provided in the dims list.
        The dims list also holds nominal values for the fixed dimensions.

        Example2:
        plot_data2d((0.5, (0.0, 1.0), (0.5, 1.0)))
          - plots for 3d problem over points (0.5, x_1, x_2)
            where x_1 in (0.0, 1.0) and x_2 in (0.5, 1.0)

        plot_data2d(((0.0, 1.0), 0.3, 0.4, (0.1, 0.3)))
          - plots for 4d problem over points (x_1, 0.3, 0.4, x_2)
            where x_1 in (0.0, 1.0) and x_2 in (0.1, 0.3)

        The min/max ranges can be subset of the corresponding domain min/max
        but should not exit the domain as the solution will not be valid.

        num_points is the number of points to take in each direction,
          keep the number even to avoid putting points right at the discontinuities
          in-between the basis functions
        '''
        assert len(dims) == self.num_dimensions, "the length of dims must match num_dimensions"
        assert num_points % 2 == 0, "num_points must be an even number"

        irange = []
        for i in range(self.num_dimensions):
            if isinstance(dims[i], (list, tuple)):
                assert len(dims[i]) == 2 or len(dims[i]) == 0, "elements of dims must be scalars or tuples/lists of len 0/2"
                irange.append(i)
            else:
                assert self.dimension_min[i] <= dims[i] and dims[i] <= self.dimension_max[i], "nominal value for dimension %d is out of bounds" %i

        assert len(irange) == 2, "there should be only 2 ranges with min/max in dims"

        mins = []
        maxs = []
        for i in irange:
            if len(dims[i]) == 2:
                for v in dims[i]:
                    assert self.dimension_min[i] <= v and v <= self.dimension_max[i], "range for dimension %d is out of bounds" %i
                mins.append(dims[i][0])
                maxs.append(dims[i][1])
            else:
                mins.append(self.dimension_min[i])
                maxs.append(self.dimension_max[i])

        x = np.linspace(mins[0] + self.eps, maxs[0] - self.eps, num_points)
        y = np.linspace(mins[1] + self.eps, maxs[1] - self.eps, num_points)

        XX, YY = np.meshgrid(x, y)

        pgrid = np.zeros((XX.size, self.num_dimensions))
        pgrid[:, irange[0]] = XX.reshape((-1,))
        pgrid[:, irange[1]] = YY.reshape((-1,))

        for i in range(self.num_dimensions):
            if not isinstance(dims[i], (list, tuple)):
                pgrid[:, i] = dims[i]

        return self.evaluate(pgrid).reshape(XX.shape), XX, YY

    def evaluate(self, points):
        '''
        points must be a 2d numpy array with shape (num_x, num_dimensions)
        where the num_x is the number of points for the evaluate command

        each row must describe a point inside the domain, but no checking will
        be performed

        returns a vector with the size num_x holding the corresponding values
        of the field
        '''
        assert len(points.shape) == 2, 'points must be a 2d numpy array with num-columns equal to self.num_dimensions'
        assert points.shape[1] == self.num_dimensions, 'points.shape[1] (%d) must be equal to self.num_dimensions (%d)' % (points.shape[1], self.num_dimensions)

        presult = np.empty((points.shape[0], ), np.float64)
        libasgard.asgard_reconstruct_solution(self.recsol,
                                              np.ctypeslib.as_ctypes(points.reshape(-1,)),
                                              points.shape[0],
                                              np.ctypeslib.as_ctypes(presult.reshape(-1,)))
        return presult

    def cell_centers(self):
        '''
        returns the geometric centers of the sparse grid cells

        do not attempt to plot at these points as they are set right on the
        the discontinuities of the basis functions and the evaluate()
        algorithm may be unstable due to rounding error
        '''
        presult = np.empty((self.num_cells, self.num_dimensions), np.float64)

        libasgard.asgard_reconstruct_cell_centers(self.recsol,
                                                  np.ctypeslib.as_ctypes(presult.reshape(-1,)))

        return presult


    def __str__(self):
        s = "title: %s\n" % self.title
        if self.subtitle != "":
            s += "   sub: %s\n" % self.subtitle
        s += "  num-dimensions: %d\n" % self.num_dimensions
        s += "  degree:         %d\n" % self.degree
        s += "  time:           %f\n" % self.time
        return s

if __name__ == "__main__":
    if len(sys.argv) < 2:
        libasgard.asgard_print_version_help()
    elif sys.argv[1] in ("-s", "-stat", "-stats", "-summary"):
        if len(sys.argv) < 3:
            print("stats summary option requires a filename")
        else:
            shot = pde_snapshot(sys.argv[2])
            print("\n", shot, shot.timer_report)
    elif not _matplotlib_found_:
        print("could not 'import matplotlib'")
        print("can only print stats-summary, use")
        print("  python3 -m asgard -s %s" % sys.argv[2])
        libasgard.asgard_print_version_help()
    else:
        shot = pde_snapshot(sys.argv[1])
        print("\n", shot)

        asgplot.title(shot.title, fontsize = 'large')

        if shot.num_dimensions == 1:
            z, x = shot.plot_data1d(((),), num_points = 256)
            asgplot.plot(x, z)
            asgplot.xlabel(shot.dimension_names[0], fontsize = 'large')
        else:
            plist = [(), ()]
            for i in range(2, shot.num_dimensions):
                plist.append(0.49 * (shot.dimension_max[i] + shot.dimension_min[i]))

            z, x, y = shot.plot_data2d(plist, num_points = 256)

            xmin = shot.dimension_min[0]
            ymin = shot.dimension_min[1]
            xmax = shot.dimension_max[0]
            ymax = shot.dimension_max[1]

            p = asgplot.imshow(z, cmap='jet', extent=[xmin, xmax, ymin, ymax])

            asgplot.colorbar(p, orientation='vertical')

            asgplot.gca().set_anchor('C')

            asgplot.xlabel(shot.dimension_names[0], fontsize='large')
            asgplot.ylabel(shot.dimension_names[1], fontsize='large')

        if len(sys.argv) > 2:
            asgplot.savefig(sys.argv[2])
        else:
            asgplot.show()

