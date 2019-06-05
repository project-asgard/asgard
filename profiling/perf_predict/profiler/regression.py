from numpy import array, polyfit, RankWarning, seterr
from warnings import catch_warnings, filterwarnings


# Represents a polynomial function on a graph
# Uses polynomial regression to get terms of the polynomial form of the function
# Also overloads the call operator for testing the function
class Function:
    def __init__(self, degree=2):
        self.degree = degree
        self.clear()

    # Clear the function's data points
    def clear(self):
        self.x = []
        self.y = []

    # Add data point to function for regression
    def add_point(self, x, y):
        self.x.append(x)
        self.y.append(y)

    # Get the function's polynomial terms as a list of floats
    def as_terms(self):
        x = array(self.x)
        y = array(self.y)
        z = polyfit(x, y, self.degree)

        return list(z)

    # Get the function as a c++ expression as a function of `var`
    def as_cpp(self, var='x'):
        result = ''
        for i, n in enumerate(self.as_terms()[::-1]):
            # If you want smaller coefficients in the C++ code, replace
            # `{n}` with `{round(n, 6)}` using any number you want in place of 6
            result += f' {n} * pow({var}, {i}) +'
        return result[1:-2]

    # Call the function obtained through regression
    def __call__(self, x):
        terms = self.as_terms()
        result = 0
        # Calculate the value of each term in the polynomial
        for exp, n in enumerate(terms[::-1]):
            result += n * (x ** exp)
        return result
