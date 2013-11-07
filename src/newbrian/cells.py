"""
Definition of cell classes for the brian module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""


import numpy
import brian
mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz
ampere = brian.amp
from pyNN.parameters import Sequence
from . import simulator


def _new_property(obj_hierarchy, attr_name, units):
    """
    Return a new property, mapping attr_name to obj_hierarchy.attr_name.

    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """
    def set(self, value):
        if obj_hierarchy:
            obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        else:
            obj = self
        setattr(obj, attr_name, value*units)
    def get(self):
        if obj_hierarchy:
            obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        else:
            obj = self
        return getattr(obj, attr_name)/units
    return property(fset=set, fget=get)


class BaseNeuronGroup(brian.NeuronGroup):

    def __init__(self, n, equations, threshold, reset=brian.NoReset(),
                 refractory=0*ms, implicit=False, **parameters):
        if "tau_refrac" in parameters:
            max_refractory = parameters["tau_refrac"].max() * ms
        else:
            max_refractory = None
        brian.NeuronGroup.__init__(self, n,
                                   model=equations,
                                   threshold=threshold,
                                   reset=reset,
                                   refractory=refractory,
                                   max_refractory=max_refractory,
                                   compile=True,
                                   clock=simulator.state.network.clock,
                                   max_delay=simulator.state.max_delay*ms,
                                   implicit=implicit,
                                   freeze=False)
        for name, value in parameters.items():
            setattr(self, name, value)
        self.initial_values = {}

    def initialize(self):
        #print("INITIALIZE: %s" % self.initial_values)
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)


class ThresholdNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.Reset(parameters.pop('v_reset'))
        refractory = parameters.pop('tau_refrac')
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset,
                                 refractory, **parameters)
        #self._variable_refractory_time = True
        #self._S0 = self._S[:, 0]

    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset    = _new_property('_resetfun', 'resetvalue', mV)

    def check_threshold(self, v):
        return v >= self.v_thresh


class BiophysicalNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.EmpiricalThreshold(threshold=-40*mV, refractory=2*ms)
        reset = 0*mV
        refractory = 0*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 implicit=True,
                                 **parameters)


class AdaptiveReset(object):

    def __init__(self, Vr=-70.6*mV, b=0.0805*nA):
        self.Vr = Vr
        self.b  = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.w[spikes] += self.b[spikes]


class AdaptiveNeuronGroup(BaseNeuronGroup):
    
    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.SimpleCustomRefractoriness(
                    AdaptiveReset(parameters.pop('v_reset'),
                                  parameters.pop('b')),
                    period=parameters['tau_refrac'])  #.max()*ms)
        refractory = None
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        #self._variable_refractory_time = True
        #self._refractory_variable = None
        #self._S0 = self._S[:, 0]

    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset    = _new_property('_resetfun.resetfun', 'Vr', mV)
    b = _new_property('_resetfun.resetfun', 'b', nA)

    def check_threshold(self, v):
        return v >= self.v_spike*mV


class IzhikevichReset(object):

    def __init__(self, Vr=-65*mV, d=0.2*mV/ms):
        self.Vr = Vr
        self.d  = d

    def __call__(self, P, spikes):
        P.v[spikes]  = self.Vr[spikes]
        P.u[spikes] += self.d[spikes]


class IzhikevichNeuronGroup(BaseNeuronGroup):
    
    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.SimpleCustomRefractoriness(
                    IzhikevichReset(parameters['v_reset'],
                                    parameters['d']),
                    period=0*ms)
        refractory = 0*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        #self._variable_refractory_time = True
        #self._refractory_variable = None
        #self._S0 = self._S[:, 0]

    v_reset    = _new_property('_resetfun.resetfun', 'Vr', mV)
    b = _new_property('_resetfun.resetfun', 'b', nA)

    def check_threshold(self, v):
        return v >= 30*mV
    

class PoissonGroup(brian.PoissonGroup):
    
    def __init__(self, n, equations, **parameters):
        for name, value in parameters.items():
            setattr(self, name, value)
        brian.PoissonGroup.__init__(self, n,
                                    rates=self.update_rates,
                                    clock=simulator.state.network.clock)

    def update_rates(self, t):
        idx = (self.start <= t) & (t <= self.start + self.duration)
        return numpy.where(idx, self.rate, 0)

    def initialize(self):
        pass


class SpikeGeneratorGroup(brian.SpikeGeneratorGroup):
    
    def __init__(self, n, equations, spike_times=None):
        """
        Note that `equations` is not used: it is simply for compatibility with
        other NeuronGroup subclasses.
        """
        spiketimes = [(i, t) for i, seq in enumerate(spike_times) for t in seq.value]
        brian.SpikeGeneratorGroup.__init__(self, n, spiketimes,
                                           clock=simulator.state.network.clock)

    def _get_spike_times(self):
        values = [list() for i in range(self.N)]
        for i, t in self.spiketimes:
            values[i].append(t)
        return numpy.array([Sequence(times) for times in values], dtype=Sequence)
    def _set_spike_times(self, spike_times, mask=None):
        if mask is not None:
            existing_times = self._get_spike_times()
            existing_times[mask] = spike_times
            spike_times = existing_times
        values = [(i, t) for i, seq in enumerate(spike_times) for t in seq.value]
        brian.SpikeGeneratorGroup.__init__(self, self.N, values, period=self.period)
    spike_times = property(fget=_get_spike_times, fset=_set_spike_times)

    def initialize(self):
        pass