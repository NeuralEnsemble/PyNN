"""
Definition of NativeElectrodeType class for NEST.
"""

import numpy
import nest
from pyNN.common import Population, PopulationView, Assembly
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.nest.simulator import state
from pyNN.nest.cells import get_defaults
from pyNN.models import BaseCurrentSource
from .conversion import make_sli_compatible


class NestCurrentSource(BaseCurrentSource):
    """Base class for a nest source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        self._device = nest.Create(self.nest_name)
        self.cell_list = []
        self.parameter_space = ParameterSpace(self.default_parameters,
                                              self.get_schema(),
                                              shape=(1,))
        if parameters:
            self.parameter_space.update(**parameters)

        self.min_delay = state.min_delay
        self.timestep = state.dt  # NoisyCurrentSource has a parameter called "dt", so use "timestep" here

    def inject_into(self, cells):
        for id in cells:
            if id.local and not id.celltype.injectable:
                raise TypeError("Can't inject current into a spike source.")
        if isinstance(cells, (Population, PopulationView, Assembly)):
            self.cell_list = [cell for cell in cells]
        else:
            self.cell_list = cells
        nest.Connect(self._device, self.cell_list, syn_spec={"delay": state.min_delay})

    def _delay_correction(self, value):
        """
        A change in a device requires a min_delay to take effect at the target
        """
        corrected = value - self.min_delay
        # set negative times to zero
        if isinstance(value, numpy.ndarray):
            corrected = numpy.where(corrected > 0, corrected, 0.0)
        else:
            corrected = max(corrected, 0.0)
        return corrected

    def record(self):
        self.i_multimeter = nest.Create('multimeter', params={'record_from': ['I'], 'interval': state.dt})
        nest.Connect(self.i_multimeter, self._device)

    def _get_data(self):
        events = nest.GetStatus(self.i_multimeter)[0]['events']
        # Similar to recording.py: NEST does not record values at
        # the zeroth time step, so we add them here.
        t_arr = numpy.insert(numpy.array(events['times']), 0, 0.0)
        i_arr = numpy.insert(numpy.array(events['I']/1000.0), 0, 0.0)
        # NEST and pyNN have different concepts of current initiation times
        # To keep this consistent across simulators, we will have current
        # initiating at the electrode at t_start and effect on cell at next dt
        # This requires padding min_delay equivalent period with 0's
        pad_length = int(self.min_delay/self.timestep)
        i_arr = numpy.insert(i_arr[:-pad_length], 0, [0]*pad_length)
        return t_arr, i_arr


def native_electrode_type(model_name):
    """
    Return a new NativeElectrodeType subclass.
    """
    assert isinstance(model_name, str)
    default_parameters, default_initial_values = get_defaults(model_name)
    return type(model_name,
               (NativeElectrodeType,),
                {'nest_name': model_name,
                 'default_parameters': default_parameters,
                 'default_initial_values': default_initial_values,
                })


# Should be usable with any NEST current generator
class NativeElectrodeType(NestCurrentSource):

    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        NestCurrentSource.__init__(self, **parameters)
        self.parameter_space.evaluate(simplify=True)
        nest.SetStatus(self._device,
                       make_sli_compatible(self.parameter_space.as_dict()))
