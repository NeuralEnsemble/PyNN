"""
Current source classes for the brian2 module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    ACSource           -- a sine modulated current.
    NoisyCurrentSource -- a Gaussian whitish noise current.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import numpy as np
from brian2 import ms, nA, Hz
from ...standardmodels import electrodes, build_translations, StandardCurrentSource
from ...parameters import ParameterSpace, Sequence
from .. import simulator

logger = logging.getLogger("PyNN")


def update_currents():
    for current_source in simulator.state.current_sources:
        current_source._update_current()


class Brian2CurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.cell_list = []
        self.indices = []
        self.previous_amplitude = 0.0
        self.running = False
        self.recording = False
        self._brian_parameters = {}
        self.i = 0
        simulator.state.current_sources.append(self)

        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)
        self._reset()

    def get_native_parameters(self):
        return ParameterSpace(self._brian_parameters)

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if isinstance(value, Sequence):
                value = value.value
            self._brian_parameters[name] = value
        self._update_state()

    def _reset(self):
        # self.i reset to 0 only at the start of a new run; not for continuation of existing runs
        if not self.running:
            self.i = 0
            self.running = True
            self.streaming = False

    def inject_into(self, cell_list):
        for cell in cell_list:
            if not cell.celltype.injectable:
                raise TypeError("Can't inject current into a spike source.")
        self.cell_list.extend(cell_list)
        for cell in cell_list:
            cell_idx = cell.parent.id_to_index(cell)
            self.indices.extend([cell_idx])

    def _update_current(self):
        """
        This is called on every time step, and updates, if necessary, the
        "i_inj" attribute of the brian2_group associated with all cells into
        which this current is being injected.

        If the current should be unchanged, nothing happens.
        """

        if self.running:
            current_time = simulator.state.t * ms
            amplitude = None
            # check if the current time corresponds to a change in
            # the injected current or streaming status
            if abs(current_time - self._times[self.i]) < (simulator.state.dt * ms / 2.0):
                if self.streamable:
                    # for streamable sources, the update times correspond to switching
                    # streaming on and off
                    self.streaming = not self.streaming
                    amplitude = 0.0 * nA
                elif not self.streaming:
                    # we have a pre-computed current waveform
                    # if we're at the last update time, the injected current should be
                    # set to zero, and we can stop making updates
                    amplitude = self._amplitudes[self.i]
                if self.i >= len(self._times) - 1:
                    self.running = False
                self.i += 1

            if self.streaming:
                # the injected current amplitude is expected to change at every
                # time-step, so we calculate it on the fly
                amplitude = self._compute_amplitude(current_time)

            if amplitude is not None:
                # cells can receive inputs from more than one current source,
                # so we change the total injected current by the difference from the
                # last value we injected from here, rather than setting the
                # absolute value
                delta = amplitude - self.previous_amplitude
                for cell, idx in zip(self.cell_list, self.indices):
                    cell.parent.brian2_group.i_inj[idx] += delta
                self.previous_amplitude = amplitude

    def record(self):
        self.recording = True


class NonStreamableCurrentSource(Brian2CurrentSource):
    streamable = False

    def _get_data(self):
        dt = simulator.state.dt * ms
        n = int(round(self._times[0] / dt))
        segments = [np.zeros(n)]
        for t1, t2, amp in zip(self._times[:-1], self._times[1:], self._amplitudes):
            n = int(round((t2 - t1) / dt))
            segments.append(amp * np.ones(n) / nA)
        n = int(round((simulator.state.t * ms - self._times[-1]) / dt))
        segments.append(self._amplitudes[-1] * np.ones(n + 1))
        amplitudes = np.hstack(segments) * nA
        times = np.arange(amplitudes.size) * simulator.state.dt * ms
        assert amplitudes.size == times.size
        return times / ms, amplitudes / nA


class StreamableCurrentSource(Brian2CurrentSource):
    streamable = True

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.streaming = False
        self.recorded_times = []
        self.recorded_amplitudes = []
        self._current_amplitude = 0.0 * nA

    def _update_state(self):
        start = self._brian_parameters["start"]
        stop = self._brian_parameters["stop"]
        self._times = [start, stop]
        if start <= simulator.state.t * ms < stop:
            self.streaming = True
        else:
            self.streaming = False

    def _get_data(self):
        dt = simulator.state.dt * ms
        len_t = int(round((simulator.state.t * ms) / dt)) + 1
        times = simulator.state.dt * ms * np.arange(len_t)
        amps = np.zeros(len_t) * nA

        start_index = int(self._brian_parameters["start"] / dt)
        len_rec = len(self.recorded_times)
        times[start_index:start_index + len_rec] = self.recorded_times
        amps[start_index:start_index + len_rec] = self.recorded_amplitudes
        return (times / ms, amps / nA)


class StepCurrentSource(NonStreamableCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes', 'amplitudes', nA),
        ('times', 'times', ms)
    )

    def _update_state(self):
        self._times, self._amplitudes = self._check_step_times(
            self._brian_parameters["times"],
            self._brian_parameters["amplitudes"],
            simulator.state.dt)
        self._brian_parameters["times"] = self._times
        self._brian_parameters["amplitudes"] = self._amplitudes

    def _check_step_times(self, times, amplitudes, resolution):
        # change resolution from ms to s; as per brian2 convention
        resolution = resolution * ms
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
        # note that we use np.flip to get the _last_ value specified for a given time step
        # e.g. if we have times = [0.41, 0.41, 0.86] and amplitudes [1, 2, 3]
        # then the times should map to [0.4, 0.9] if dt = 0.1 and the amplitudes should be [2, 3]
        step_times, index = np.unique(np.flip(times), return_index=True)
        step_amplitudes = np.flip(amplitudes)[index]
        return step_times, step_amplitudes


class DCSource(NonStreamableCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms)
    )

    def _update_state(self):
        start = self._brian_parameters["start"]
        stop = self._brian_parameters["stop"]
        amplitude = self._brian_parameters["amplitude"]

        if start == 0 * ms:
            self._times = [start, stop]
            self._amplitudes = [amplitude, 0.0]
        else:
            self._times = [0.0 * ms, start, stop]
            self._amplitudes = [0.0 * nA, amplitude, 0.0 * nA]
        # ensures proper handling of changes in parameters on the fly
        if start < simulator.state.t * ms < stop:
            self._times.insert(-1, simulator.state.t * ms)
            self._amplitudes.insert(-1, amplitude)
            if (start == 0 and self.i == 2) or (start != 0 and self.i == 3):
                self.i -= 1


class ACSource(StreamableCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms),
        ('frequency', 'frequency', Hz),
        ('offset', 'offset', nA),
        ('phase', 'phase')
    )

    def _compute_amplitude(self, time):
        # Note: Brian2 uses seconds as unit of time;
        #       frequency is specified in Hz; thus no conversion required
        offset = self._brian_parameters["offset"]
        waveform_amplitude = self._brian_parameters["amplitude"]
        delta_t = time - self._brian_parameters["start"]
        freq = self._brian_parameters["frequency"]
        phase = self._brian_parameters["phase"]
        current_amplitude = offset + waveform_amplitude * np.sin(
            delta_t * 2 * np.pi * freq + 2 * np.pi * phase / 360)
        if self.recording:
            self.recorded_times.append(time)
            self.recorded_amplitudes.append(current_amplitude)
        return current_amplitude


class NoisyCurrentSource(StreamableCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__
    # not streamable unless dt parameter is same as simulator dt?

    translations = build_translations(
        ('mean', 'mean', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms),
        ('stdev', 'stdev', nA),
        ('dt', 'dt', ms)
    )

    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.j = 0
        if self._brian_parameters["dt"] != simulator.state.dt * ms:
            self.change_at = int(self._brian_parameters["dt"] / simulator.state.dt / ms)
        else:
            self.change_at = 1

    def _compute_amplitude(self, time):
        if self.j == 0:
            self._current_amplitude = (
                self._brian_parameters["mean"]
                + self._brian_parameters["stdev"] * np.random.randn()
            )
        self.j += 1
        if self.j == self.change_at:
            self.j = 0
        if self.recording:
            self.recorded_times.append(time)
            self.recorded_amplitudes.append(self._current_amplitude)
        return self._current_amplitude
