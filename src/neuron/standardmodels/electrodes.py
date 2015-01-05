"""
Current source classes for the neuron module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.
    ACSource           -- a sine modulated current.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from neuron import h
import numpy
from pyNN.standardmodels import electrodes, build_translations, StandardCurrentSource
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.neuron import simulator


_current_sources = []  # if a CurrentSource is created but not assigned to a variable,
                       # it will not persist, so we store a reference here


class NeuronCurrentSource(StandardCurrentSource):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self, **parameters):
        self._devices    = []
        self.cell_list   = []
        self._amplitudes = None
        self._times      = None
        self._h_iclamps  = {}
        parameter_space = ParameterSpace(self.default_parameters,
                                         self.get_schema(),
                                         shape=(1,))
        parameter_space.update(**parameters)
        parameter_space = self.translate(parameter_space)
        self.set_native_parameters(parameter_space)
        _current_sources.append(self)

    @property
    def _h_amplitudes(self):
        if self._amplitudes == None:
            if isinstance(self.amplitudes, Sequence):
                self._amplitudes = h.Vector(self.amplitudes.value)
            else:
                self._amplitudes = h.Vector(self.amplitudes)
        return self._amplitudes

    @property
    def _h_times(self):
        if self._times == None:
            if isinstance(self.times, Sequence):
                self._times = h.Vector(self.times.value)
            else:
                self._times = h.Vector(self.times)
        return self._times

    def _reset(self):
        if self._is_computed:
            self._amplitudes = None
            self._times      = None
            self._generate()
        for iclamp in self._h_iclamps.values():
            self._update_iclamp(iclamp)

    def _update_iclamp(self, iclamp):
        if not self._is_playable:
            iclamp.delay = max(0, self.start - simulator.state.t)
            iclamp.dur   = self.stop-self.start
            iclamp.amp   = self.amplitude

        if self._is_playable:
            iclamp.delay = 0.0
            iclamp.dur   = 1e12
            iclamp.amp   = 0.0
            self._h_amplitudes.play(iclamp._ref_amp, self._h_times)

    def set_native_parameters(self, parameters):
        parameters.evaluate(simplify=True)
        for name, value in parameters.items():
            if isinstance(value, Sequence):  # this shouldn't be necessary, but seems to prevent a segfault
                value = value.value
            object.__setattr__(self, name, value)
        self._reset()

    def get_native_parameters(self):
        return ParameterSpace(dict((k, self.__getattribute__(k)) for k in self.get_native_names()))

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        for id in cells:
            if id.local:
                if not id.celltype.injectable:
                    raise TypeError("Can't inject current into a spike source.")
                if not (id in self._h_iclamps):
                    self.cell_list += [id]
                    self._h_iclamps[id] = h.IClamp(0.5, sec=id._cell.source_section)
                    self._devices.append(self._h_iclamps[id])
                self._update_iclamp(self._h_iclamps[id])

    def _record(self):
        self.itrace = h.Vector()
        self.itrace.record(self._devices[0]._ref_i)
        self.record_times = h.Vector()
        self.record_times.record(h._ref_t)

    def _get_data(self):
        return numpy.array((self.record_times, self.itrace))


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
        ## Not efficient at all... Is there a way to have those vectors computed on the fly ?
        ## Otherwise should have a buffer mechanism
        self.times      = numpy.arange(self.start, self.stop+simulator.state.dt, simulator.state.dt)
        tmp             = numpy.arange(0, self.stop - self.start, simulator.state.dt)
        self.amplitudes = self.offset + self.amplitude * numpy.sin(tmp*2*numpy.pi*self.frequency/1000. + 2*numpy.pi*self.phase/360)
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
        ## Not efficient at all... Is there a way to have those vectors computed on the fly ?
        ## Otherwise should have a buffer mechanism
        self.times      = numpy.arange(self.start, self.stop+simulator.state.dt, simulator.state.dt)
        tmp             = numpy.arange(0, self.stop - self.start, simulator.state.dt)
        self.amplitudes = self.mean + (self.stdev*self.dt)*numpy.random.randn(len(tmp))
        self.amplitudes[-1] = 0.0
