"""
Current source classes for the brian module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    ACSource           -- a sine modulated current.
    NoisyCurrentSource -- a Gaussian whitish noise current.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import numpy
from brian import ms, nA, Hz, network_operation, amp as ampere
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from .. import simulator

logger = logging.getLogger("PyNN")

current_sources = []


@network_operation(when='start')
def update_currents():
    global current_sources
    for current_source in current_sources:
        current_source._update_current()


class BrianCurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        super(StandardCurrentSource, self).__init__(**parameters)
        global current_sources
        self.cell_list = []
        self.indices   = []
        current_sources.append(self)
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if isinstance(value, Sequence):
                value = value.value
            object.__setattr__(self, name, value)
        self._reset()

    def _reset(self):
        self.i       = 0
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
        if self.running and simulator.state.t >= self.times[self.i]*1e3:
            for cell, idx in zip(self.cell_list, self.indices):
                if not self._is_playable:
                    cell.parent.brian_group.i_inj[idx] = self.amplitudes[self.i] * ampere
                else:
                    cell.parent.brian_group.i_inj[idx] = self._compute(self.times[self.i]) * ampere
            self.i += 1
            if self.i >= len(self.times):
                self.running = False


class StepCurrentSource(BrianCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes', nA),
        ('times',       'times',      ms)
    )
    _is_computed = False
    _is_playable = False


class ACSource(BrianCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude', nA),
        ('start',      'start',     ms),
        ('stop',       'stop',      ms),
        ('frequency',  'frequency', Hz),
        ('offset',     'offset',    nA),
        ('phase',      'phase',     1)
    )
    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        BrianCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        self.times = numpy.arange(self.start, self.stop, simulator.state.dt)

    def _compute(self, time):
        return self.amplitude * numpy.sin(time*2*numpy.pi*self.frequency/1000. + 2*numpy.pi*self.phase/360)


class DCSource(BrianCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude', nA),
        ('start',      'start',     ms),
        ('stop',       'stop',      ms)
    )
    _is_computed = True
    _is_playable = False

    def __init__(self, **parameters):
        BrianCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        if self.start == 0:
            self.times      = [self.start, self.stop]
            self.amplitudes = [self.amplitude, 0.0]
        else:
            self.times      = [0.0, self.start, self.stop]
            self.amplitudes = [0.0, self.amplitude, 0.0]


class NoisyCurrentSource(BrianCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean',  'mean',    nA),
        ('start', 'start',   ms),
        ('stop',  'stop',    ms),
        ('stdev', 'stdev',   nA),
        ('dt',    'dt',      ms)
    )
    _is_computed = True
    _is_playable = True

    def __init__(self, **parameters):
        BrianCurrentSource.__init__(self, **parameters)
        self._generate()

    def _generate(self):
        self.times = numpy.arange(self.start, self.stop, simulator.state.dt)

    def _compute(self, time):
        return self.mean + (self.stdev*self.dt)*numpy.random.randn()
