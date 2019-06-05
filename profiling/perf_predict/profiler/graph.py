# import numpy as np
import matplotlib.pyplot as plt
from regression import Function
from spreadsheet import MemorySheet


# Used to plot regression.Functions or lambda functions
class Graph:
    def __init__(self, title="ASGarD", max_x=8):
        # These will contain the lists of x positions and y positions of the different functions
        self.xs = []
        self.ys = []
        self.fmts = []  # formatting for each list of points
        self.max_x = max_x
        self.title = title

    # Adds a list of x and y points to be plotted, with optional formatting
    def add_points(self, xs, ys, fmt=''):
        self.xs.append(xs)
        self.ys.append(ys)
        self.fmts.append(fmt)

    # Plot the graph when youre done
    def plot(self):
        plt.title(self.title)
        for (x, y, fmt) in zip(self.xs, self.ys, self.fmts):
            plt.plot(x, y, fmt)

        plt.show()

    # Takes a function and adds it to the graph
    def add_fn(self, fn, steps=20):
        x_points = []
        y_points = []
        for i in range(steps):
            x = i * self.max_x / steps
            x_points.append(x)
            y_points.append(fn(x))

        self.add_points(x_points, y_points)
        if isinstance(fn, Function):
            self.add_points(list(fn.x), list(fn.y), '-o')

    # Takes a memory sheet and adds the actual memory usage to the graph
    def add_mem_sheet(self, sheet):
        for level in range(1, MemorySheet.MAX_LEVELS):
            x_points = []
            y_points = []
            for degree in range(2, MemorySheet.MAX_DEGREES):
                x_points.append(degree)
                y_points.append(sheet.get_diff(level, degree))
            self.add_points(x_points, y_points, '-o')

        # if isinstance(fn, Function):
        #     self.add_points(list(fn.x), list(fn.y), '-o')
