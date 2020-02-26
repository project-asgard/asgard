from predictor import Predictor, PDE


# This class represents the memory predictor for asgard, over all pdes
class MemPredictor(Predictor):
    def __init__(self, pdes=[]):
        super().__init__(name='total_mem_usage', pdes=pdes)


# This struct represents a function that predicts memory usage for a given PDE
class MemPDE(PDE):
    def __init__(self, pde='continuity_1', definition=''):
        super().__init__(pde, definition, name=f'{pde}_MB')
