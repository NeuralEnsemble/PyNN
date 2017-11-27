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
from pyNN.nest.electrodes import NestCurrentSource


class NestStandardCurrentSource(NestCurrentSource, StandardCurrentSource):
    """Base class for a nest source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        NestCurrentSource.__init__(self, **parameters)
        self.phase_given = 0.0  # required for PR #502
        native_parameters = self.translate(self.parameter_space)
        self.set_native_parameters(native_parameters)

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


class DCSource(NestStandardCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude', 1000.),
        ('start',      'start'),
        ('stop',       'stop')
    )
    nest_name = 'dc_generator'


class ACSource(NestStandardCurrentSource, electrodes.ACSource):
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


class StepCurrentSource(NestStandardCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitude_values', 1000.),
        ('times',       'amplitude_times')
    )
    nest_name = 'step_current_generator'


class NoisyCurrentSource(NestStandardCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean',  'mean', 1000.),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'std', 1000.),
        ('dt',    'dt')
    )
    nest_name = 'noise_generator'
