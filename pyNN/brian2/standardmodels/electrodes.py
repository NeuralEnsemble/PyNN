"""
Current source classes for the brian2 module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    ACSource           -- a sine modulated current.
    NoisyCurrentSource -- a Gaussian whitish noise current.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import numpy as np
import brian2
from brian2 import ms, second, nA, amp, Hz, NetworkOperation, amp as ampere
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.brian2 import simulator

logger = logging.getLogger("PyNN")


def update_currents():
    for current_source in simulator.state.current_sources:
        current_source._update_current()


class Brian2CurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        super(StandardCurrentSource, self).__init__(**parameters)
        self.cell_list = []
        self.indices = []
        self.prev_amp_dict = {}
        self.running = False
        simulator.state.current_sources.append(self)
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)

    def _check_step_times(self, times, amplitudes, resolution):
        # change resolution from ms to s; as per brian2 convention
        resolution = resolution*1e-3
        # ensure that all time stamps are non-negative
        if not (times >= 0.0).all():
            raise ValueError("Step current cannot accept negative timestamps.")
        # ensure that times provided are of strictly increasing magnitudes
        dt_times = np.diff(times)
        if not all(dt_times > 0.0):
            raise ValueError("Step current timestamps should be monotonically increasing.")
        # map timestamps to actual simulation time instants based on specified dt
        times = self._round_timestamp(times, resolution)
        # remove duplicate timestamps, and corresponding amplitudes, after mapping
        step_times = []
        step_amplitudes = []
        for ts0, amp0, ts1 in zip(times, amplitudes, times[1:]):
            if ts0 != ts1:
                step_times.append(ts0)
                step_amplitudes.append(amp0)
        step_times.append(times[-1])
        step_amplitudes.append(amplitudes[-1])
        return step_times, step_amplitudes

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if name == "amplitudes":  # key used only by StepCurrentSource
                step_times = parameters["times"].value
                step_amplitudes = parameters["amplitudes"].value
                step_times, step_amplitudes = self._check_step_times(
                    step_times, step_amplitudes, simulator.state.dt)
                parameters["times"].value = step_times
                parameters["amplitudes"].value = step_amplitudes
            if isinstance(value, Sequence):
                value = value.value
            object.__setattr__(self, name, value)
        self._reset()

    def _reset(self):
        # self.i reset to 0 only at the start of a new run; not for continuation of existing runs
        if not hasattr(self, 'running') or self.running == False:
            self.i = 0
            self.running = True
        if self._is_computed:
            self._generate()

    def inject_into(self, cell_list):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        for cell in cell_list:
            if not cell.celltype.injectable:
                raise TypeError("Can't inject current into a spike source.")
        self.cell_list.extend(cell_list)
        for cell in cell_list:
            cell_idx = cell.parent.id_to_index(cell)
            self.indices.extend([cell_idx])
            self.prev_amp_dict[cell_idx] = 0.0

    def _update_current(self):
        # check if current timestamp is within dt/2 of target time; Brian2 uses seconds as unit of time
        if self.running and abs(simulator.state.t - self.times[self.i] * 1e3) < (simulator.state.dt/2.0):
            for cell, idx in zip(self.cell_list, self.indices):
                if not self._is_playable:
                    cell.parent.brian2_group.i_inj[idx] += (
                        self.amplitudes[self.i] - self.prev_amp_dict[idx]) * ampere
                    self.prev_amp_dict[idx] = self.amplitudes[self.i]
                else:
                    amp_val = self._compute(self.times[self.i])
                    self.amplitudes = np.append(self.amplitudes, amp_val)
                    cell.parent.brian2_group.i_inj[idx] += (amp_val -
                                                            self.prev_amp_dict[idx]) * ampere
                    self.prev_amp_dict[idx] = amp_val  # * ampere
            self.i += 1
            if self.i >= len(self.times):
                self.running = False
                if self._is_playable:
                    # ensure that currents are set to 0 after t_stop
                    for cell, idx in zip(self.cell_list, self.indices):
                        cell.parent.brian2_group.i_inj[idx] -= self.prev_amp_dict[idx] * ampere

    def record(self):
        pass

    def _get_data(self):
        def find_nearest(array, value):
            array = np.asarray(array)
            return (np.abs(array - value)).argmin()

        len_t = int(round((simulator.state.t * 1e-3) / (simulator.state.dt * 1e-3))) + 1
        times = np.array([(i * simulator.state.dt * 1e-3) for i in range(len_t)])
        amps = np.array([0.0] * len_t)

        for idx, [t1, t2] in enumerate(zip(self.times, self.times[1:])):
            if t2 < simulator.state.t * 1e-3:
                idx1 = find_nearest(times, t1)
                idx2 = find_nearest(times, t2)
                amps[idx1:idx2] = [self.amplitudes[idx]] * len(amps[idx1:idx2])
                if idx == len(self.times)-2:
                    if not self._is_playable and not self._is_computed:
                        amps[idx2:] = [self.amplitudes[idx+1]] * len(amps[idx2:])
            else:
                if t1 < simulator.state.t * 1e-3:
                    idx1 = find_nearest(times, t1)
                    amps[idx1:] = [self.amplitudes[idx]] * len(amps[idx1:])
                break
        return (times * second / ms, amps * amp / nA)


class StepCurrentSource(Brian2CurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes', 'amplitudes', nA),
        ('times', 'times', ms)
    )
    _is_computed = False
    _is_playable = False


class ACSource(Brian2CurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms),
        ('frequency', 'frequency', Hz),
        ('offset', 'offset', nA),
        ('phase', 'phase', 1)
    )
    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        Brian2CurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        # Note: Brian2 uses seconds as unit of time
        temp_num_t = int(round(((self.stop + simulator.state.dt * 1e-3) -
                                self.start) / (simulator.state.dt * 1e-3)))
        self.times = np.array([self.start + (i * simulator.state.dt * 1e-3)
                                  for i in range(temp_num_t)])
        self.amplitudes = np.zeros(0)

    def _compute(self, time):
        # Note: Brian2 uses seconds as unit of time; frequency is specified in Hz; thus no conversion required
        return self.offset + self.amplitude * np.sin((time-self.start) * 2 * np.pi * self.frequency + 2 * np.pi * self.phase / 360)


class DCSource(Brian2CurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms)
    )
    _is_computed = True
    _is_playable = False

    def __init__(self, **parameters):
        Brian2CurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        if self.start == 0:
            self.times = [self.start, self.stop]
            self.amplitudes = [self.amplitude, 0.0]
        else:
            self.times = [0.0, self.start, self.stop]
            self.amplitudes = [0.0, self.amplitude, 0.0]
        # ensures proper handling of changes in parameters on the fly
        if self.start < simulator.state.t*1e-3 < self.stop:
            self.times.insert(-1, simulator.state.t*1e-3)
            self.amplitudes.insert(-1, self.amplitude)
            if (self.start == 0 and self.i == 2) or (self.start != 0 and self.i == 3):
                self.i -= 1


class NoisyCurrentSource(Brian2CurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean', 'mean', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms),
        ('stdev', 'stdev', nA),
        ('dt', 'dt', ms)
    )
    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        Brian2CurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        temp_num_t = int(round((self.stop - self.start) / max(self.dt, simulator.state.dt * 1e-3)))
        self.times = np.array(
            [self.start + (i * max(self.dt, simulator.state.dt * 1e-3)) for i in range(temp_num_t)])
        self.times = np.append(self.times, self.stop)
        self.amplitudes = np.zeros(0)

    def _compute(self, time):
        return self.mean + self.stdev * np.random.randn()
