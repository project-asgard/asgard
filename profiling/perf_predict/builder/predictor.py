from os.path import basename


# This is the base class for all predictors
# Adds a level of abstraction so that it is simpler to add new features
class Predictor:
    # Takes a name and list of pdes
    # the name is the name of the resulting output function
    def __init__(self, name, pdes=[]):
        self.pdes = pdes
        self.name = name

    # Add a pde to the memory predictor
    def add_pde(self, pde):
        self.pdes.append(pde)

    # Generates the function that will return the mem usage for
    # the given PDE enum value, the level, and the degree
    def to_function(self, level='level', degree='degree'):
        switch = '\n  switch (pde) {\n'
        for pde in self.pdes:
            switch += f'    case {pde.to_enum()}: return {pde.to_call(level, degree)}; break;\n'
        switch += '    default: return std::make_pair("", 0);\n  }\n'

        return rf'std::pair<std::string, double> {self.name}(PDE_opts pde, int {level}, int {degree}) {{{switch}}}'

    # Creates the header containing the declarations of all the PDE
    # predictor functions and the intermediate_mem_usage functions
    def to_header(self):
        result = 'std::pair<std::string, double> {self.name}(PDE_opts, int, int);\n\n'
        for pde in self.pdes:
            result += f'{pde.to_declaration()}\n'
        return result

    # Generates the cpp file containing the function
    # definitions for the header file
    def to_cpp(self, header=None):
        result = f'#include <algorithm>\n#include <utility>\n#include <iostream>\n#include "{header}"\n\n' if header else ''
        for pde in self.pdes:
            result += pde.to_definition() + '\n\n'
        result += '\n\n\n' + self.to_function() + '\n\n'
        return result


# This is the base class for functions that predict
# some form of performance for a given pde. It's used
# by a predictor object to output code.
class PDE:
    # pde is the enum name of the pde
    # definition is the function definition of the predictor function
    # name is the name of the predictor function
    def __init__(self, pde='continuity_1', definition='', name=None):
        self.pde = pde
        self.name = name
        self.definition = definition

    # Get PDE memory predictor from the output header file from the profiler
    def from_header(self, filename, pde=None):
        pde = pde if pde else basename(filename).split('.')[0]
        with open(filename) as header:
            definition = header.read()
            header.close()

        return type(self)(pde=pde, definition=definition)

    # Get the enumeration of the PDE
    def to_enum(self): return f'PDE_opts::{self.pde}'

    # Get the function call to get the pde memory
    def to_call(self, level='level', degree='degree'):
        return f'{self.name}({level}, {degree})'

    # Return the PDE memory predictor as a function declaration
    def to_declaration(self):
        return f'std::pair<std::string, double> {self.name}(int, int);'

    # Return the PDE memory predictor as a function definition
    def to_definition(self):
        return f'{self.definition}'
