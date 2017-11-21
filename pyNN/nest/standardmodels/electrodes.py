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

    def record(self):
        self.i_multimeter = nest.Create('multimeter', params={'record_from': ['I'], 'interval' :state.dt})
        nest.Connect(self.i_multimeter, self._device)

    def get_data(self):
        events = nest.GetStatus(self.i_multimeter)[0]['events']
        # Similar to recording.py: NEST does not record values at
        # the zeroth time step, so we add them here.
        t_arr = numpy.insert(numpy.array(events['times']), 0, 0.0)
        i_arr = numpy.insert(numpy.array(events['I']/1000.0), 0, 0.0)
        # NEST and pyNN have different concepts of current initiation times
        # To keep this consistent across simulators, we will have current
        # initiating at the electrode at t_start and effect on cell at next dt
        # This requires padding min_delay equivalent period with 0's
        pad_length = int(state.min_delay/state.dt)
        i_arr = numpy.insert(i_arr[:-pad_length], 0, [0]*pad_length)
        return t_arr, i_arr


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
