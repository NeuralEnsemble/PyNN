"""
Current source classes for the neuron module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from neuron import h
import numpy as np
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.neuron import simulator


class NeuronCurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        self._devices = []
        self.cell_list = []
        self._amplitudes = None
        self._times = None
        self._h_iclamps = {}
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)
        simulator.state.current_sources.append(self)

    @property
    def _h_amplitudes(self):
        if self._amplitudes is None:
            if isinstance(self.amplitudes, Sequence):
                self._amplitudes = h.Vector(self.amplitudes.value)
            else:
                self._amplitudes = h.Vector(self.amplitudes)
        return self._amplitudes

    @property
    def _h_times(self):
        if self._times is None:
            if isinstance(self.times, Sequence):
                self._times = h.Vector(self.times.value)
            else:
                self._times = h.Vector(self.times)
        return self._times

    def _reset(self):
        if self._is_computed:
            self._amplitudes = None
            self._times = None
            self._generate()
        for iclamp in self._h_iclamps.values():
            self._update_iclamp(iclamp, 0.0)    # send tstop = 0.0 on _reset()

    def _update_iclamp(self, iclamp, tstop):
        if not self._is_playable:
            iclamp.delay = self.start
            iclamp.dur = self.stop - self.start
            iclamp.amp = self.amplitude

        if self._is_playable:
            iclamp.delay = 0.0
            iclamp.dur = 1e12
            iclamp.amp = 0.0

            # check exists only for StepCurrentSource (_is_playable = True, _is_computed = False)
            # t_stop should be part of the time sequence to handle repeated runs
            if not self._is_computed and tstop not in self._h_times.to_python():
                ind = self._h_times.indwhere(">=", tstop)
                if ind == -1:   # tstop beyond last specified time instant
                    ind = self._h_times.size()
                if ind == 0.0:    # tstop before first specified time instant
                    amp_val = 0.0
                else:
                    amp_val = self._h_amplitudes.x[int(ind)-1]
                self._h_times.insrt(ind, tstop)
                self._h_amplitudes.insrt(ind, amp_val)

            self._h_amplitudes.play(iclamp._ref_amp, self._h_times)

    def _check_step_times(self, times, amplitudes, resolution):
        # ensure that all time stamps are non-negative
        if not (times >= 0.0).all():
            raise ValueError("Step current cannot accept negative timestamps.")
        # ensure that times provided are of strictly increasing magnitudes
        dt_times = np.diff(times)
        if not all(dt_times > 0.0):
            raise ValueError("Step current timestamps should be monotonically increasing.")
        # map timestamps to actual simulation time instants based on specified dt
        for ind in range(len(times)):
            times[ind] = self._round_timestamp(times[ind], resolution)
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
                # this shouldn't be necessary, but seems to prevent a segfault
                value = value.value
            object.__setattr__(self, name, value)
        self._reset()

    def get_native_parameters(self):
        return ParameterSpace(dict((k, self.__getattribute__(k)) for k in self.get_native_names()))

    def inject_into(self, cells):
        for id in cells:
            if id.local:
                if not id.celltype.injectable:
                    raise TypeError("Can't inject current into a spike source.")
                if not (id in self._h_iclamps):
                    self.cell_list += [id]
                    self._h_iclamps[id] = h.IClamp(0.5, sec=id._cell.source_section)
                    self._devices.append(self._h_iclamps[id])

    def record(self):
        self.itrace = h.Vector()
        self.itrace.record(self._devices[0]._ref_i)
        self.recorded_times = h.Vector()
        self.recorded_times.record(h._ref_t)

    def _get_data(self):
        # NEURON and pyNN have different concepts of current initiation times
        # To keep this consistent across simulators, pyNN will have current
        # initiating at the electrode at t_start and effect on cell at next dt.
        # This requires removing the first element from the current Vector
        # as NEURON computes the currents one time step later. The vector length
        # is compensated by repeating the last recorded value of current.
        t_arr = np.array(self.recorded_times)
        i_arr = np.array(self.itrace)[1:]
        i_arr = np.append(i_arr, i_arr[-1])
        return (t_arr, i_arr)


class DCSource(NeuronCurrentSource, electrodes.DCSource):

    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )

    _is_playable = False
    _is_computed = False


class StepCurrentSource(NeuronCurrentSource, electrodes.StepCurrentSource):

    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )

    _is_playable = True
    _is_computed = False

    def _generate(self):
        pass


class ACSource(NeuronCurrentSource, electrodes.ACSource):

    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop'),
        ('frequency',  'frequency'),
        ('offset',     'offset'),
        ('phase',      'phase')
    )

    _is_playable = True
    _is_computed = True

    def __init__(self, **parameters):
        NeuronCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        # Not efficient at all... Is there a way to have those vectors computed on the fly ?
        # Otherwise should have a buffer mechanism
        temp_num_t = int(round(
            ((self.stop + simulator.state.dt) - self.start) / simulator.state.dt
        ))
        tmp = simulator.state.dt * np.arange(temp_num_t)
        self.times = tmp + self.start
        self.amplitudes = self.offset + self.amplitude * np.sin(
            tmp * 2 * np.pi * self.frequency / 1000. + 2 * np.pi * self.phase / 360
        )
        self.amplitudes[-1] = 0.0


class NoisyCurrentSource(NeuronCurrentSource, electrodes.NoisyCurrentSource):

    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean',  'mean'),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'stdev'),
        ('dt',    'dt')
    )

    _is_playable = True
    _is_computed = True

    def __init__(self, **parameters):
        NeuronCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        # Not efficient at all... Is there a way to have those vectors computed on the fly ?
        # Otherwise should have a buffer mechanism
        temp_num_t = int(round((self.stop - self.start) / max(self.dt, simulator.state.dt)))
        self.times = self.start + max(self.dt, simulator.state.dt) * np.arange(temp_num_t)
        self.times = np.append(self.times, self.stop)
        self.amplitudes = self.mean + self.stdev * np.random.randn(len(self.times))
        self.amplitudes[-1] = 0.0
