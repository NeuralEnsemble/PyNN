"""
Current source classes for the nest module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.


:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import nest
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.common import Population, PopulationView, Assembly
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.nest.simulator import state


class NestCurrentSource(StandardCurrentSource):
    """Base class for a nest source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        self._device = nest.Create(self.nest_name)
        self.cell_list = []
        self.phase_given = 0.0  # required for PR #502
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
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
        corrected = value - state.min_delay
        # set negative times to zero
        if isinstance(value, numpy.ndarray):
            corrected = numpy.where(corrected > 0, corrected, 0.0)
        else:
            corrected = max(corrected, 0.0)
        return corrected

    def _phase_correction(self, start, freq, phase):
        """
        Fixes #497 (PR #502)
        Tweaks the value of phase supplied to NEST ACSource
        so as to remain consistent with other simulators
        """
        phase_fix = ( (phase*numpy.pi/180) - (2*numpy.pi*freq*start/1000)) * 180/numpy.pi
        phase_fix.shape = (1)
        phase_fix = phase_fix.evaluate()[0]
        nest.SetStatus(self._device, {'phase': phase_fix})

    def _round_timestamp(self, value, resolution):
        # subtraction by 1e-12 to match NEURON working
        return round ((float(value)-1e-12)/ resolution) * resolution

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
                if key == "start" and type(self).__name__ == "ACSource":
                    self._phase_correction(self.start, self.frequency, self.phase_given)
            elif key == "frequency":
                nest.SetStatus(self._device, {key: value})
                self._phase_correction(self.start, self.frequency, self.phase_given)
            elif key == "phase":
                self.phase_given = value
                self._phase_correction(self.start, self.frequency, self.phase_given)
            elif not key == "amplitude_times":
                nest.SetStatus(self._device, {key: value})

    def get_native_parameters(self):
        all_params = nest.GetStatus(self._device)[0]
        return ParameterSpace(dict((k, v) for k, v in all_params.items()
                                   if k in self.get_native_names()))


class DCSource(NestCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude', 1000.),
        ('start',      'start'),
        ('stop',       'stop')
    )
    nest_name = 'dc_generator'


class ACSource(NestCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude', 1000.),
        ('start',      'start'),
        ('stop',       'stop'),
        ('frequency',  'frequency'),
        ('offset',     'offset',    1000.),
        ('phase',      'phase')
    )
    nest_name = 'ac_generator'


class StepCurrentSource(NestCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitude_values', 1000.),
        ('times',       'amplitude_times')
    )
    nest_name = 'step_current_generator'


class NoisyCurrentSource(NestCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean',  'mean', 1000.),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'std', 1000.),
        ('dt',    'dt')
    )
    nest_name = 'noise_generator'


# Creation of native_electrode_type
# See Issue 506
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
              'synaptic_elements']
    default_params = {}
    default_initial_values = {}
    for name, value in defaults.items():
       if name in variables:
            default_initial_values[name] = value
    return default_params, default_initial_values

def get_receptor_types(model_name):
    return list(nest.GetDefaults(model_name).get("receptor_types", ('excitatory', 'inhibitory')))

def native_electrode_type(model_name):
    """
    Return a new NativeElectrodeType subclass.
    """
    assert isinstance(model_name, str)
    default_parameters, default_initial_values = get_defaults(model_name)
    receptor_types = get_receptor_types(model_name)
    return type(model_name,
               (NativeElectrodeType,),
                {'nest_model': model_name,
                 'default_parameters': default_parameters,
                 'default_initial_values': default_initial_values,
                 'injectable': ("V_m" in default_initial_values),
                 'nest_name': {"on_grid": model_name, "off_grid": model_name}
                 })


class NativeElectrodeType(NestCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NativeElectrodeType.__doc__

    translations = build_translations(
        ('mean',  'mean', 1000.),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'std', 1000.),
        ('dt',    'dt')
    )

    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        self._device = nest.Create(self) # For NativeElectrodeType
        self.cell_list = []
        self.phase_given = 0.0  # required for PR #502
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space) # In link with translations = build_translations()
        self.set_native_parameters(parameter_space)

    def get_native_electrode_type(self):
        # Call to the function native_electrode_type
        return nest.GetDefaults(self.nest_model)["native_electrode_type"][name]

    def get_receptor_type(self, name):
        return nest.GetDefaults(self.nest_model)["receptor_types"][name]

    def inject_into(self, cells):
        # Call to the function inject_into from NestCurrentSource
        super(NativeElectrodeType,self).inject_into([cells])

    def get_recordables(model_name):
        return [sl.name for sl in nest.GetDefaults(model_name).get("recordables", [])]

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

