"""
Current source classes for the nemo module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: electrodes.py 991 2011-09-30 13:05:02Z apdavison $
"""

import numpy
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource, ModelNotAvailable
from pyNN.random import NumpyRNG, NativeRNG
from pyNN.common import Population, PopulationView, Assembly

# should really use the StandardModel machinery to allow reverse translations

class NemoCurrentSource(StandardCurrentSource):
    """Base class for a nest source of current to be injected into a neuron."""

    def __init__(self, parameters):    
        super(StandardCurrentSource, self).__init__(parameters)
        self.set_native_parameters(parameters)

    def inject_into(self, cell_list):
        pass

    def set_native_parameters(self, parameters):
        parameters = self.translate(parameters)
        for key, value in parameters.items():
            self.parameters[key] = value

    def get_native_parameters(self):    
        return self.parameters
    

class DCSource(ModelNotAvailable):
    pass
    
class ACSource(ModelNotAvailable):
    pass

class StepCurrentSource(ModelNotAvailable):
    pass

class NoisyCurrentSource(ModelNotAvailable):
    pass

