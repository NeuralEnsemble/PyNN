"""
Definition of cell classes for the brian module.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
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

from pyNN import core
from pyNN.parameters import Sequence
from pyNN.brian import simulator

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


class AdaptiveReset(object):

    def __init__(self, Vr=-70.6*mV, b=0.0805*nA):
        self.Vr = Vr
        self.b  = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.w[spikes] += self.b[spikes]


class IzhikevichReset(object):

    def __init__(self, Vr= -65 * mV, d=0.2 * mV/ms):
        self.Vr = Vr
        self.d  = d

    def __call__(self, P, spikes):
        P.v[spikes]  = self.Vr[spikes]
        P.u[spikes] += self.d[spikes]


class BaseNeuronGroup(brian.NeuronGroup):

    def __init__(self, n, equations, threshold, reset, refractory,
                 implicit=False, **parameters):
        try:
            clock = simulator.state.simclock
            max_delay = simulator.state.max_delay*ms
        except AttributeError:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        if "tau_refrac" in parameters:
            max_refractory = parameters["tau_refrac"].max() * ms
        else:
            max_refractory = None
        brian.NeuronGroup.__init__(self, n,
                                   model=equations,
                                   threshold=threshold,
                                   reset=reset,
                                   refractory=refractory,
                                   max_refractory = max_refractory,
                                   compile=True,
                                   clock=simulator.state.simclock,
                                   max_delay=simulator.state.max_delay*ms,
                                   implicit=implicit,
                                   freeze=False)
        for name, value in parameters.items():
            setattr(self, name, value)
        self.initial_values = {}

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)


class BiophysicalNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.EmpiricalThreshold(threshold=-40*mV, refractory=2*ms)
        reset = 0*mV
        refractory = 0*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 implicit=True,
                                 **parameters)
        self._S0 = self._S[:, 0]


class ThresholdNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.Reset(parameters['v_reset']*mV)
        refractory = parameters['tau_refrac']*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        self._variable_refractory_time = True
        self._S0 = self._S[:, 0]

    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset    = _new_property('_resetfun', 'resetvalue', mV)

    def check_threshold(self, v):
        return v >= self.v_thresh*mV


class AdaptiveNeuronGroup(BaseNeuronGroup):
    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.SimpleCustomRefractoriness(
                    AdaptiveReset(parameters['v_reset']* mV,
                                  parameters['b']*ampere),
                    period=parameters['tau_refrac'].max()*ms)
        refractory = None
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        self._variable_refractory_time = True
        self._refractory_variable = None
        self._S0 = self._S[:, 0]

    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset    = _new_property('_resetfun.resetfun', 'Vr', mV)
    b = _new_property('_resetfun.resetfun', 'b', nA)

    def check_threshold(self, v):
        return v >= self.v_spike*mV


class IzhikevichNeuronGroup(BaseNeuronGroup):
    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.SimpleCustomRefractoriness(
                    IzhikevichReset(parameters['v_reset']* mV,
                                  parameters['d']),
                    period=parameters['tau_refrac'].max()*ms)
        refractory = None
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        self._variable_refractory_time = True
        self._refractory_variable = None
        self._S0 = self._S[:, 0]

    tau_refrac = _new_property('', '_refractory_array', ms)
    v_reset    = _new_property('_resetfun.resetfun', 'Vr', mV)
    b = _new_property('_resetfun.resetfun', 'b', nA)

    def check_threshold(self, v):
        return v >= 30*mV


class PoissonGroupWithDelays(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.PoissonThreshold()
        reset = brian.NoReset()
        refractory = 0*ms
        BaseNeuronGroup.__init__(self, n,
                                 brian.LazyStateUpdater(),
                                 threshold, reset, refractory,
                                 **parameters)
        self.initialize()

    def update_rates(self, t):
        """
        Acts as a function of time for the PoissonGroup, while storing the
        parameters for later retrieval.
        """
        idx = (self.start <= t) & (t <= self.start + self.duration)
        return numpy.where(idx, self.rate, 0)

    def update(self):
        self._S[0, :] = self.update_rates(self.clock.t)
        brian.NeuronGroup.update(self)

    def initialize(self):
        self._S0[0] = self.update_rates(self.clock.t)


class MultipleSpikeGeneratorGroupWithDelays(BaseNeuronGroup):

    def __init__(self, n, equations, spiketimes=None):
        threshold = brian.directcontrol.MultipleSpikeGeneratorThreshold(
                                               [st.value for st in spiketimes])
        reset = brian.NoReset()
        refractory = 0*ms
        BaseNeuronGroup.__init__(self, n,
                                 brian.LazyStateUpdater(),
                                 threshold, reset, refractory,
                                 spiketimes=spiketimes)

    def _get_spiketimes(self):
        return self._threshold.spiketimes
    def _set_spiketimes(self, spiketimes):
        if core.is_listlike(spiketimes):
            assert len(spiketimes) == len(self), "spiketimes (length %d) must contain as many iterables as there are cells in the group (%d)." % (len(spiketimes), len(self))
            assert isinstance(spiketimes[0], Sequence)
            self._threshold.set_spike_times([st.value for st in spiketimes])
        elif isinstance(spiketimes, Sequence):
            self._threshold.set_spike_times([spiketimes.value for i in range(len(self))])
        else:
            raise Exception()
    spiketimes = property(fget=_get_spiketimes, fset=_set_spiketimes)

    def reinit(self):
        brian.NeuroGroup.reinit(self)
        self._threshold.reinit()

    def initialize(self):
        pass

