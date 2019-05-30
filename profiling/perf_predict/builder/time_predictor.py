from predictor import Predictor, PDE


# This class represents the memory predictor for asgard, over all pdes
class TimePredictor(Predictor):
    def __init__(self, pdes=[]):
        super().__init__(name='expected_time', pdes=pdes)


# This struct represents a function that predicts compute time for a given PDE
class TimePDE(PDE):
    def __init__(self, pde='continuity_1', definition=''):
        super().__init__(pde, definition, name=f'{pde}_seconds')
