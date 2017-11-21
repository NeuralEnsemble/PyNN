"""
Definition of NativeElectrodeType class for NEST.
"""

import numpy
import nest
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.common import Population, PopulationView, Assembly
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.nest.simulator import state
from pyNN.nest.standardmodels.electrodes import NestCurrentSource
from pyNN.models import BaseCellType
from pyNN.nest.cells import NativeCellType
from pyNN.nest.cells import get_receptor_types
from pyNN.nest.cells import get_recordables
from . import conversion


def get_defaults(model_name):
    valid_types = (int, float, Sequence)
    defaults = nest.GetDefaults(model_name)
    variables = defaults.get('recordables', [])
    ignore = ['archiver_length', 'available', 'Ca', 'capacity', 'elementsize',
              'frozen', 'instantiations', 'local', 'model', 'needs_prelim_update',
              'recordables', 'state', 't_spike', 'tau_minus', 'tau_minus_triplet',
              'thread', 'vp', 'receptor_types', 'events', 'global_id',
              'element_type', 'type', 'type_id', 'has_connections', 'n_synapses',
              'thread_local_id', 'node_uses_wfr', 'supports_precise_spikes',
              'synaptic_elements'
                ]
    default_params = {}
    default_initial_values = {}
    for name, value in defaults.items():
        if name in variables:
            default_initial_values[name] = value
    return default_params, default_initial_values
    
# Native model non standard
def native_electrode_type(model_name):
    """
    Return a new NativeElectrodeType subclass.
    """
    assert isinstance(model_name, str)
    default_parameters, default_initial_values = get_defaults(model_name)  
    receptor_types = get_receptor_types(model_name) # get_receptor_types imported from nest.cells.py
    return type(model_name,
               (NativeElectrodeType,),
                {'nest_model': model_name,
                 'default_parameters': default_parameters,
                 'default_initial_values': default_initial_values,
                 'injectable': ("V_m" in default_initial_values),
                 'nest_name': {"on_grid": model_name, "off_grid": model_name},
                })
    

# Should be usable with any NEST current generator
class NativeElectrodeType(NestCurrentSource):
    
    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        self._device = nest.Create(self)
        self.cell_list = []
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        self.set_native_parameters(parameter_space)

    def get_native_electrode_type(self):
        # Call to the function native_electrode_type
        return nest.GetDefaults(self.nest_model)["native_electrode_type"][name]

    def get_receptor_type(self, name):
        return nest.GetDefaults(self.nest_model)["receptor_types"][name]

    def inject_into(self, cells):
        # Call to the function inject_into from NestCurrentSource
        super(NativeElectrodeType,self).inject_into([cells])

    def _generate(self):
        self.times = numpy.arange(self.start, self.stop, max(self.dt, simulator.state.dt))
        self.times = numpy.append(self.times, self.stop)
        self.amplitudes = self.mean + self.stdev * numpy.random.randn(len(self.times))
        self.amplitudes[-1] = 0.0

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for key, value in parameters.items():    
            if key == "amplitude_values":
                assert isinstance(value, Sequence)
                times = self._delay_correction(parameters["amplitude_times"].value)
                amplitudes = value.value
                ctr = next((i for i,v in enumerate(times) if v > state.dt), len(times)) - 1
                if ctr >= 0:
                    times[ctr] = state.dt
                    times = times[ctr:]
                    amplitudes = amplitudes[ctr:]
                for ind in range(len(times)):
                    times[ind] = self._round_timestamp(times[ind], state.dt)
                nest.SetStatus(self._device, {key: amplitudes,
                                              'amplitude_times': times})
            elif key in ("start", "stop"):
                nest.SetStatus(self._device, {key: self._delay_correction(value)})
                #For NativeElectrodeType class
                if key == "start" and type(self).__name__ == "NativeElectrodeType":
                    self._phase_correction(self.start, self.frequency, self.phase_given)
            elif key == "frequency":
                nest.SetStatus(self._device, {key: value})
                self._phase_correction(self.start, self.frequency, self.phase_given)
            elif key == "phase":
                self.phase_given = value
                self._phase_correction(self.start, self.frequency, self.phase_given)
           
            elif not key == "amplitude_times":
                nest.SetStatus(self._device, {key: value})
                self.default_parameters[key] = value
            