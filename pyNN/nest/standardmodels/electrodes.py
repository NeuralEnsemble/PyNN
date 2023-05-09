"""
Current source classes for the nest module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.


:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy as np
import nest
from ...standardmodels import electrodes, build_translations, StandardCurrentSource
from ...common import Population, PopulationView, Assembly
from ...parameters import ParameterSpace, Sequence
from ..simulator import state
from ..electrodes import NestCurrentSource


class NestStandardCurrentSource(NestCurrentSource, StandardCurrentSource):
    """Base class for a nest source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        NestCurrentSource.__init__(self, **parameters)
        self.phase_given = 0.0  # required for PR #502
        native_parameters = self.translate(self.parameter_space)
        self.set_native_parameters(native_parameters)

    def inject_into(self, cells):
        for id in cells:
            if id.local and not id.celltype.injectable:
                raise TypeError("Can't inject current into a spike source.")
        if isinstance(cells, (Population, PopulationView, Assembly)):
            self.cell_list = cells.node_collection
        else:
            self.cell_list = nest.NodeCollection(sorted(cells))
        nest.Connect(self._device, self.cell_list, syn_spec={"delay": state.min_delay})

    def _delay_correction(self, value):
        """
        A change in a device requires a min_delay to take effect at the target
        """
        corrected = value - self.min_delay
        # set negative times to zero
        if isinstance(value, np.ndarray):
            corrected = np.where(corrected > 0, corrected, 0.0)
        else:
            corrected = max(corrected, 0.0)
        return corrected

    def _phase_correction(self, start, freq, phase):
        """
        Fixes #497 (PR #502)
        Tweaks the value of phase supplied to NEST ACSource
        so as to remain consistent with other simulators
        """
        phase_fix = ((phase*np.pi/180) - (2*np.pi*freq*start/1000)) * 180/np.pi
        phase_fix.shape = (1)
        phase_fix = phase_fix.evaluate()[0]
        nest.SetStatus(self._device, {'phase': phase_fix})

    def _check_step_times(self, times, amplitudes, resolution):
        # ensure that all time stamps are non-negative
        if np.min(times) < 0:
            raise ValueError("Step current cannot accept negative timestamps.")
        # ensure that times provided are of strictly increasing magnitudes
        if len(times) > 1 and np.min(np.diff(times)) <= 0:
            raise ValueError("Step current timestamps should be monotonically increasing.")
        # NEST specific: subtract min_delay from times (set to 0.0, if result is negative)
        times = self._delay_correction(times)
        # find the last element <= dt (we find >dt and then go one element back)
        # this corresponds to the first timestamp that can be used by NEST for current injection
        ctr = np.searchsorted(times, resolution, side="right") - 1
        if ctr >= 0:
            times[ctr] = resolution
            times = times[ctr:]
            amplitudes = amplitudes[ctr:]
        # map timestamps to actual simulation time instants based on specified dt
        # for ind in range(len(times)):
        #    times[ind] = self._round_timestamp(times[ind], resolution)
        times = self._round_timestamp(times, resolution)
        # remove duplicate timestamps, and corresponding amplitudes, after mapping
        step_times, step_indices = np.unique(times[::-1], return_index=True)
        step_times = step_times.tolist()
        step_indices = len(times)-step_indices-1
        step_amplitudes = amplitudes[step_indices]  # [amplitudes[i] for i in step_indices]
        return step_times, step_amplitudes

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for key, value in parameters.items():
            if key == "amplitude_values":
                assert isinstance(value, Sequence)
                step_times = parameters["amplitude_times"].value
                step_amplitudes = parameters["amplitude_values"].value

                step_times, step_amplitudes = self._check_step_times(
                    step_times, step_amplitudes, self.timestep)
                parameters["amplitude_times"].value = step_times
                parameters["amplitude_values"].value = step_amplitudes
                nest.SetStatus(self._device, {key: step_amplitudes,
                                              'amplitude_times': step_times})
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
