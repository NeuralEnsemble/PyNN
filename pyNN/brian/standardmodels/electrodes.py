"""
Current source classes for the brian module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    ACSource           -- a sine modulated current.
    NoisyCurrentSource -- a Gaussian whitish noise current.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import numpy
import brian
from brian import ms, nA, Hz, network_operation, amp as ampere
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from .. import simulator

logger = logging.getLogger("PyNN")


@network_operation(when='start')
def update_currents():
    for current_source in simulator.state.current_sources:
        current_source._update_current()


class BrianCurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        super(StandardCurrentSource, self).__init__(**parameters)
        self.cell_list = []
        self.indices = []
        simulator.state.current_sources.append(self)
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)

    def _check_step_times(self, times, amplitudes, resolution):
        # change resolution from ms to s; as per brian convention
        resolution = resolution*1e-3
        # ensure that all time stamps are non-negative
        if not (times >= 0.0).all():
            raise ValueError("Step current cannot accept negative timestamps.")
        # ensure that times provided are of strictly increasing magnitudes
        dt_times = numpy.diff(times)
        if not all(dt_times>0.0):
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
            if name == "amplitudes": # key used only by StepCurrentSource
                step_times = parameters["times"].value
                step_amplitudes = parameters["amplitudes"].value
                step_times, step_amplitudes = self._check_step_times(step_times, step_amplitudes, simulator.state.dt)
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
            self.indices.extend([cell.parent.id_to_index(cell)])

    def _update_current(self):
        # check if current timestamp is within dt/2 of target time; Brian uses seconds as unit of time
        if self.running and abs(simulator.state.t - self.times[self.i] * 1e3) < (simulator.state.dt/2.0):
            for cell, idx in zip(self.cell_list, self.indices):
                if not self._is_playable:
                    cell.parent.brian_group.i_inj[idx] = self.amplitudes[self.i] * ampere
                else:
                    cell.parent.brian_group.i_inj[idx] = self._compute(self.times[self.i]) * ampere
            self.i += 1
            if self.i >= len(self.times):
                self.running = False
                if self._is_playable:
                    # ensure that currents are set to 0 after t_stop
                    for cell, idx in zip(self.cell_list, self.indices):
                        cell.parent.brian_group.i_inj[idx] = 0

    def record(self):
        self.i_state_monitor = brian.StateMonitor(self.cell_list[0].parent.brian_group[self.indices[0]], 'i_inj', record=0, when='start')
        simulator.state.network.add(self.i_state_monitor)

    def _get_data(self):
        # code based on brian/recording.py:_get_all_signals()
        # because we use `when='start'`, we need to add the
        # value at the end of the final time step.
        device = self.i_state_monitor
        current_t_value = device.P.state_('t')[device.record]
        current_i_value = device.P.state_(device.varname)[device.record]
        t_all_values = numpy.append(device.times, current_t_value)
        i_all_values = numpy.append(device._values, current_i_value)
        return (t_all_values / ms, i_all_values / nA)


class StepCurrentSource(BrianCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes', 'amplitudes', nA),
        ('times', 'times', ms)
    )
    _is_computed = False
    _is_playable = False


class ACSource(BrianCurrentSource, electrodes.ACSource):
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
        BrianCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        # Note: Brian uses seconds as unit of time
        temp_num_t = len(numpy.arange(self.start, self.stop + simulator.state.dt * 1e-3, simulator.state.dt * 1e-3))
        self.times = numpy.array([self.start+(i*simulator.state.dt*1e-3) for i in range(temp_num_t)])

    def _compute(self, time):
        # Note: Brian uses seconds as unit of time; frequency is specified in Hz; thus no conversion required
        return self.offset + self.amplitude * numpy.sin((time-self.start) * 2 * numpy.pi * self.frequency + 2 * numpy.pi * self.phase / 360)


class DCSource(BrianCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude', 'amplitude', nA),
        ('start', 'start', ms),
        ('stop', 'stop', ms)
    )
    _is_computed = True
    _is_playable = False

    def __init__(self, **parameters):
        BrianCurrentSource.__init__(self, **parameters)
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
            if (self.start==0 and self.i==2) or (self.start!=0 and self.i==3):
                self.i -= 1


class NoisyCurrentSource(BrianCurrentSource, electrodes.NoisyCurrentSource):
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
        BrianCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        temp_num_t = len(numpy.arange(self.start, self.stop, max(self.dt, simulator.state.dt * 1e-3)))
        self.times = numpy.array([self.start+(i*max(self.dt, simulator.state.dt * 1e-3)) for i in range(temp_num_t)])
        self.times = numpy.append(self.times, self.stop)

    def _compute(self, time):
        return self.mean + self.stdev * numpy.random.randn()
