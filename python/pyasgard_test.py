
import unittest
import numpy as np
import os

import asgard

def basis_linear0(x):
    return np.ones(x.shape)

def basis_linear1(x):
    return 2.0 * np.sqrt(3.0) * x - np.sqrt(3.0)

def basis_linear2(x):
    return ((x < 0.5) * np.sqrt(3.0) * (1.0 - 4.0 * x)
            + (x > 0.5) * np.sqrt(3.0) * (-3.0 + 4.0 * x))

def basis_linear3(x):
    return (x < 0.5) * (-1.0 + 6.0 * x) + (x > 0.5) * (-5.0 + 6.0 * x)

def basis_linear4(x):
    return ((x < 0.5) * np.sqrt(2.0) * ((x < 0.25) * np.sqrt(3.0) * (1.0 - 8.0 * x)
                                        + (x > 0.25) * np.sqrt(3.0) * (-3.0 + 8.0 * x)))

def basis_linear5(x):
    return ((x < 0.5) * np.sqrt(2.0) * ((x < 0.25) * (-1.0 + 12.0 * x)
                                        + (x > 0.25) * (-5.0 + 12.0 * x)))

def basis_linear6(x):
    return ((x > 0.5) * np.sqrt(2.0) * ((x < 0.75) * np.sqrt(3.0) * (5.0 - 8.0 * x)
                                        + (x > 0.75) * np.sqrt(3.0) * (-7.0 + 8.0 * x)))

def basis_linear7(x):
    return ((x > 0.5) * np.sqrt(2.0) * ((x < 0.75) * (-7.0 + 12.0 * x)
                                        + (x > 0.75) * (-11.0 + 12.0 * x)))

def continuity_1_exact(x, t):
    return np.cos(2.0 * np.pi * x) * np.sin(t)

def continuity_2_exact(x, y, t):
    return np.cos(np.pi * x) * np.sin(2.0 * np.pi * y) * np.sin(2.0 * t)

def continuity_3_exact(x, y, z, t):
    return continuity_2_exact(x, y, t) * np.cos(2.0 * np.pi * z / 3.0)

class asgard_reconstruction_tests(unittest.TestCase):
    def almost_equal(self, x, y, message):
        verbose = True
        digits  = 13
        np.testing.assert_almost_equal(x, y, digits, message, verbose)

    def onedim_match_basis(self, ibasis, basis):
        assert ibasis < 8, "1d test is set for 4 cells or 8 basis functions"
        num_cells = 4
        dmin = np.array((0.0,))
        dmax = np.array((1.0,))
        cells = np.array((0, 0, 1, 0, 2, 0, 2, 1), np.int32)
        state = np.zeros((2 * num_cells,), np.float64)
        state[ibasis] = 1.0
        libasgard = asgard.libasgard
        pntr = libasgard.asgard_make_dreconstruct_solution(
                1, num_cells, np.ctypeslib.as_ctypes(cells),
                1, np.ctypeslib.as_ctypes(state))

        libasgard.asgard_reconstruct_solution_setbounds(
            pntr, np.ctypeslib.as_ctypes(dmin), np.ctypeslib.as_ctypes(dmax))

        x = np.linspace(0.0001, 0.9999, 8)
        y = np.zeros(x.shape)

        yref = basis(x)
        libasgard.asgard_reconstruct_solution(
            pntr, np.ctypeslib.as_ctypes(x), x.size, np.ctypeslib.as_ctypes(y))

        self.almost_equal(y, yref, "exact basis reconstruction: ibasis = %d" % ibasis)

        libasgard.asgard_pydelete_reconstruct_solution(pntr)

    def test1d_basis_reconstruct(self):
        print("\ntesting 1d (linear) basis reconstruction")
        self.onedim_match_basis(0, basis_linear0)
        self.onedim_match_basis(1, basis_linear1)

        self.onedim_match_basis(2, basis_linear2)
        self.onedim_match_basis(3, basis_linear3)

        self.onedim_match_basis(4, basis_linear4)
        self.onedim_match_basis(5, basis_linear5)

        self.onedim_match_basis(6, basis_linear6)
        self.onedim_match_basis(7, basis_linear7)

    def twodim_match_basis(self, ibasis, basis):
        assert ibasis < 24, "2d test is set for 6 cells or 24 basis functions"
        num_cells = 6
        dmin = np.array((0.0, 0.0))
        dmax = np.array((1.0, 1.0))
        cells = np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0), np.int32)
        state = np.zeros((4 * num_cells,), np.float64)
        state[ibasis] = 1.0
        libasgard = asgard.libasgard
        pntr = libasgard.asgard_make_dreconstruct_solution(
                2, num_cells, np.ctypeslib.as_ctypes(cells),
                1, np.ctypeslib.as_ctypes(state))

        libasgard.asgard_reconstruct_solution_setbounds(
            pntr, np.ctypeslib.as_ctypes(dmin), np.ctypeslib.as_ctypes(dmax))

        x = np.linspace(0.0001, 0.9999, 8)
        y = np.zeros(x.shape)

        x = np.column_stack((x, 0.0001 * np.ones(x.shape)))

        yref = basis(x)
        libasgard.asgard_reconstruct_solution(
            pntr, np.ctypeslib.as_ctypes(x.reshape(-1,)), x.shape[0], np.ctypeslib.as_ctypes(y))

        self.almost_equal(y, yref, "exact basis reconstruction: ibasis = %d" % ibasis)

        x = np.linspace(0.0001, 0.9999, 8)
        y = np.zeros(x.shape)

        x = np.column_stack((0.0001 * np.ones(x.shape), x))

        yref = basis(x)
        libasgard.asgard_reconstruct_solution(
            pntr, np.ctypeslib.as_ctypes(x.reshape(-1,)), x.shape[0], np.ctypeslib.as_ctypes(y))

        self.almost_equal(y, yref, "exact basis reconstruction: ibasis = %d" % ibasis)

        libasgard.asgard_pydelete_reconstruct_solution(pntr)

    def test2d_basis_reconstruct(self):
        print("\ntesting 2d (linear) basis reconstruction")
        # cell (0, 0)
        self.twodim_match_basis(0, lambda x : basis_linear0(x[:,0]) * basis_linear0(x[:,1]))
        self.twodim_match_basis(1, lambda x : basis_linear0(x[:,0]) * basis_linear1(x[:,1]))
        self.twodim_match_basis(2, lambda x : basis_linear1(x[:,0]) * basis_linear0(x[:,1]))
        self.twodim_match_basis(3, lambda x : basis_linear1(x[:,0]) * basis_linear1(x[:,1]))
        # cell (0, 1)
        self.twodim_match_basis(4, lambda x : basis_linear0(x[:,0]) * basis_linear2(x[:,1]))
        self.twodim_match_basis(5, lambda x : basis_linear0(x[:,0]) * basis_linear3(x[:,1]))
        self.twodim_match_basis(6, lambda x : basis_linear1(x[:,0]) * basis_linear2(x[:,1]))
        self.twodim_match_basis(7, lambda x : basis_linear1(x[:,0]) * basis_linear3(x[:,1]))
        # cell (0, 2)
        self.twodim_match_basis( 8, lambda x : basis_linear0(x[:,0]) * basis_linear4(x[:,1]))
        self.twodim_match_basis( 9, lambda x : basis_linear0(x[:,0]) * basis_linear5(x[:,1]))
        self.twodim_match_basis(10, lambda x : basis_linear1(x[:,0]) * basis_linear4(x[:,1]))
        self.twodim_match_basis(11, lambda x : basis_linear1(x[:,0]) * basis_linear5(x[:,1]))

    # simple IO test
    def test_simple1d(self):
        print("\ntesting 1d plot")
        tols = (1.E-4, 1.E-6, 1.E-8, 1.E-9)
        levs = (6, 6, 4, 4)
        for degree in range(4):
            os.system("./asgard -p continuity_1 -d %d -l 6 -dt 0.005 -of _test_plot.h5 1>/dev/null" % degree)

            self.assertTrue(os.path.isfile("_test_plot.h5"), "failed to run continuity_1")

            snapshot = asgard.pde_snapshot("_test_plot.h5")

            gold_dimension_min = np.array([-1.0, ])
            gold_dimension_max = np.array([1.0, ])
            self.almost_equal(snapshot.dimension_min, gold_dimension_min,
                              "mismatch in dimension_min")
            self.almost_equal(snapshot.dimension_max, gold_dimension_max,
                              "mismatch in dimension_max")

            self.assertEqual(snapshot.num_dimensions, 1, "mismatch in the number of dimensions")
            self.assertEqual(snapshot.num_cells, 64, "mismatch in the number of cells")

            z, x = snapshot.plot_data1d([[-1.0, 1.0],], num_points = 128)

            # exact solution for the continuity_1 example
            h = continuity_1_exact(x, snapshot.time)

            err = np.sum(np.abs(h - z) ** 2) / x.size

            self.assertLessEqual(err, tols[degree], "mismatchin continuity_1")

    def test_simple2d(self):
        print("\ntesting 2d plot")
        os.system("./asgard -p continuity_2 -d 1 -l 6 -w 10 -dt 0.00001 1>/dev/null")

        self.assertTrue(os.path.isfile("asgard_wavelet_10.h5"), "failed to run continuity_2")

        snapshot = asgard.pde_snapshot("asgard_wavelet_10.h5")

        self.assertEqual(snapshot.num_dimensions, 2, "mismatch in the number of dimensions")
        self.assertEqual(snapshot.num_cells, 256, "mismatch in the number of cells")

        gold_dimension_min = np.array([-1.0, -2.0])
        gold_dimension_max = np.array([1.0, 2.0])
        self.almost_equal(snapshot.dimension_min, gold_dimension_min,
                          "mismatch in dimension_min")
        self.almost_equal(snapshot.dimension_max, gold_dimension_max,
                          "mismatch in dimension_max")

        z, x, y = snapshot.plot_data2d([[-1.0, 1.0], [-2.0, 2.0]], num_points = 128)

        # exact solution for the continuity_2 example
        h = continuity_2_exact(x, y, snapshot.time)

        err = np.sum(np.abs(h - z) ** 2) / x.size
        self.assertLessEqual(err, 1.E-9, "mismatchin continuity_2")

        # test the 1d plot inside of 2d domain
        z, x = snapshot.plot_data1d(((-1.0, 1.0), 0.1), num_points = 128)
        h = continuity_2_exact(x, 0.1 * np.ones(x.shape), snapshot.time)

        err = np.sum(np.abs(h - z) ** 2) / x.size
        self.assertLessEqual(err, 1.E-10, "mismatchin continuity_2")

    def test_simple3d(self):
        print("\ntesting 3d plot (longer test)")
        os.system("./asgard -p continuity_3 -d 1 -l 8 -w 10 -dt 0.0001 1>/dev/null")

        self.assertTrue(os.path.isfile("asgard_wavelet_10.h5"), "failed to run continuity_3")

        snapshot = asgard.pde_snapshot("asgard_wavelet_10.h5")

        self.assertEqual(snapshot.num_dimensions, 3, "mismatch in the number of dimensions")
        self.assertEqual(snapshot.num_cells, 4096, "mismatch in the number of cells")

        gold_dimension_min = np.array([-1.0, -2.0, -3.0])
        gold_dimension_max = np.array([1.0, 2.0, 3.0])
        self.almost_equal(snapshot.dimension_min, gold_dimension_min,
                          "mismatch in dimension_min")
        self.almost_equal(snapshot.dimension_max, gold_dimension_max,
                          "mismatch in dimension_max")

        z, x, y = snapshot.plot_data2d([0.1, [-2.0, 2.0], [-3.0, 3.0]], num_points = 128)

        # exact solution for the continuity_2 example
        h = continuity_3_exact(0.1 * np.ones(x.shape), x, y, snapshot.time)

        err = np.sum(np.abs(h - z) ** 2) / x.size
        self.assertLessEqual(err, 1.E-8, "mismatchin continuity_3")

    def test_cellcenters(self):
        print("\ntesting cell centers")
        os.system("./asgard -p continuity_2 -d 1 -l 2 -n 0 -of cells.h5 -dt 0.01 1>/dev/null")

        self.assertTrue(os.path.isfile("cells.h5"), "failed to generate output for cell centers")

        snapshot = asgard.pde_snapshot("cells.h5")

        cells = snapshot.cell_centers()

        # wavelet basis level 0 and 1 have the same centers at 0
        # cells indexes are (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 0), (3, 0)
        ref_cells = np.array(((0, 0), (0, 0), (0, -1), (0, 1), (0, 0), (0, 0), (-0.5, 0.0), (0.5, 0.0)))
        self.almost_equal(cells, ref_cells, "mismatch in the reported grid")

if __name__ == '__main__':
    unittest.main()
