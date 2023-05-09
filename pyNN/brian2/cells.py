"""
Definition of cell classes for the brian2 module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from functools import reduce
import numpy as np
import brian2

from ..parameters import Sequence, simplify
from ..core import is_listlike
from .. import errors
from . import simulator

mV = brian2.mV
ms = brian2.ms
nA = brian2.nA
uS = brian2.uS
nS = brian2.nS
Hz = brian2.Hz
nF = brian2.nF
ampere = brian2.amp
second = brian2.second


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
        setattr(obj, attr_name, value * units)

    def get(self):
        if obj_hierarchy:
            obj = reduce(getattr, [self] + obj_hierarchy.split('.'))  # ( getattr, ...)
        else:
            obj = self
        return getattr(obj, attr_name) / units
    return property(fset=set, fget=get)


class BaseNeuronGroup(brian2.NeuronGroup):

    def __init__(self, n, equations, threshold, reset=-20 * mV,
                 refractory=0 * ms, method=None, **parameters):
        if "tau_refrac" in parameters:
            if parameters["tau_refrac"].min() != parameters["tau_refrac"].max():
                raise Exception("Non-homogeneous refractory period not yet supported")
            refractory = parameters["tau_refrac"].min()
        if method is None:
            method = ('exact', 'euler', 'heun')  # Brian 2 default
        brian2.NeuronGroup.__init__(self, n,
                                    model=equations,
                                    threshold=threshold,
                                    reset=reset,
                                    refractory=refractory,
                                    method=method,
                                    clock=simulator.state.network.clock)
        for name, value in parameters.items():

            if not hasattr(self, name):
                self.add_attribute(name)
            else:
                setattr(self, name, value)
        # self._S0 = self._S[:, 0]  # store parameter values in case of reset.
                # TODO: update this when parameters are modified
                # TODO: Brian2 does not have _S0a
        self.add_attribute('initial_values')
        self.initial_values = {}

    def _get_tau_refrac(self):
        return self._refractory

    def _set_tau_refrac(self, value):
        self._refractory = simplify(value)

    tau_refrac = property(fget=_get_tau_refrac, fset=_set_tau_refrac)

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)


class ThresholdNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v > v_thresh'
        reset = 'v = v_reset'
        equations += '''v_thresh : volt (constant)
                        v_reset  : volt (constant)'''
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset=reset, **parameters)


class BiophysicalNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v > -40*mV'
        refractory = False
        reset = None
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold, reset=reset, refractory=refractory,
                                 **parameters)
        # implicit=True, compile=False,


class AdaptiveReset(object):

    def __init__(self, Vr=-70.7 * mV, b=0.0805 * nA):
        self.Vr = Vr
        self.b = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.w[spikes] += self.b[spikes]


class AdaptiveNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        thresh = parameters["v_thresh"][0]
        Vcut = thresh + parameters["delta_T"][0] * 5
        threshold = 'v > {}*mV'.format(Vcut / mV)
        self._resetvalue = parameters.pop('v_reset')[0]
        self._bvalue = parameters.pop('b')[0]
        self._refracvalue = parameters.pop('tau_refrac')[0]
        reset = 'v = {}*mV; w+={}*amp'.format(self._resetvalue / mV, self._bvalue / ampere)
        refractory = self._refracvalue
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold, reset=reset, refractory=refractory,
                                 method="rk2", **parameters)

    @property
    def v_reset(self):
        return self._resetvalue

    @property
    def b(self):
        return self._bvalue

    @property
    def tau_refrac(self):
        return self._refractory

    @tau_refrac.setter
    def tau_refrac(self, tau_refrac_value):
        self._refractory = tau_refrac_value

    @v_reset.setter
    def v_reset(self, resetvalue):
        self.event_codes['spike'] = 'v = {}*mV'.format(resetvalue / mV)


class AdaptiveReset2(object):
    def __init__(self, v_reset, q_r, q_s):
        self.v_reset = v_reset
        self.q_r = q_r
        self.q_s = q_s

    def __call__(self, P, spikes):
        P.v[spikes] = self.v_reset[spikes]
        P.g_r[spikes] += self.q_r[spikes]
        P.g_s[spikes] += self.q_s[spikes]


class AdaptiveNeuronGroup2(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v >= {}*mV'.format(parameters["v_thresh"][0] / mV)
        self._resetvalue = parameters.pop('v_reset')[0]
        self._q_rvalue = parameters.pop('q_r')[0] * 10**9
        self._q_svalue = parameters.pop('q_s')[0] * 10**9
        self._refracvalue = parameters.pop('tau_refrac')[0]
        reset = 'v = {}*mV; g_r+= {}*nS; g_s+={}*nS'.format(
            self._resetvalue / mV, self._q_rvalue / nS, self._q_svalue / nS)
        refractory = self._refracvalue
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset=reset, refractory=refractory,
                                 method="rk2", **parameters)

    @property
    def v_reset(self):
        return self._resetvalue

    @property
    def q_r(self):
        return self._q_rvalue

    @property
    def q_s(self):
        return self._q_svalue

    @property
    def tau_refrac(self):
        return self._refractory

    @tau_refrac.setter
    def tau_refrac(self, tau_refrac_value):
        self._refractory = tau_refrac_value

    @v_reset.setter
    def v_reset(self, resetvalue):
        self.event_codes['spike'] = 'v = {}*mV'.format(resetvalue / mV)


# The below can be replaced by
# reset = '''v = v_reset
#            u += d'''


class IzhikevichReset(object):

    def __init__(self, Vr=-65 * mV, d=0.2 * mV / ms):
        self.Vr = Vr
        self.d = d

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr[spikes]
        P.u[spikes] += self.d[spikes]


class IzhikevichNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = 'v >= 30*mV'
        self._resetvalue = parameters.pop('v_reset')[0]
        self._dvalue = parameters.pop('d')[0]
        reset = 'v = {}*mV; u+={}*mV/ms'.format(self._resetvalue / mV, self._dvalue / (mV/ms))
        refractory = 0 * ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold=threshold, reset=reset, refractory=refractory,
                                 **parameters)

    @property
    def v_reset(self):
        return self._resetvalue

    @property
    def d(self):
        return self._dvalue

    @v_reset.setter
    def v_reset(self, resetvalue):
        self.event_codes['spike'] = 'v = {}*mV'.format(resetvalue / mV)


class PoissonGroup(brian2.PoissonGroup):

    def __init__(self, n, equations, **parameters):
        self.start_time = simplify(parameters["start_time"])
        self.firing_rate = parameters["firing_rate"]
        self.duration = simplify(parameters["duration"])

        brian2.PoissonGroup.__init__(self, n,
                                     rates=self.firing_rate,
                                     clock=simulator.state.network.clock)
        if is_listlike(self.start_time):
            self.variables.add_array('start_time', size=n, dimensions=second.dim)
        else:
            self.variables.add_constant('start_time', value=float(
                self.start_time), dimensions=second.dim)
        if is_listlike(self.duration) or is_listlike(self.start_time):
            self.variables.add_array('end_time', size=n, dimensions=second.dim)
            self.end_time = self.start_time + self.duration
        else:
            self.variables.add_constant('end_time', value=float(
                self.start_time + self.duration), dimensions=second.dim)
        self.events = {'spike': '(t >= start_time) and (t <= end_time) and (rand() < rates * dt)'}

    def initialize(self):
        pass


class SpikeGeneratorGroup(brian2.SpikeGeneratorGroup):

    def __init__(self, n, equations, spike_time_sequences=None):
        """
        Note that `equations` is not used: it is simply for compatibility with
        other NeuronGroup subclasses.
        """
        assert spike_time_sequences.size == n
        self._check_spike_times(spike_time_sequences)
        indices, times = self._convert_sequences_to_arrays(spike_time_sequences)
        brian2.SpikeGeneratorGroup.__init__(self, n, indices=indices, times=times)

    def _convert_sequences_to_arrays(self, spike_time_sequences):
        times = np.concatenate([seq.value for seq in spike_time_sequences])
        indices = np.concatenate([i * np.ones(seq.value.size)
                                  for i, seq in enumerate(spike_time_sequences)])
        return indices, times * second
        # todo: try to push the multiplication by seconds back into the translation step.
        #       note that the scaling from ms to seconds does take place during translation

    def _get_spike_time_sequences(self):
        # todo: might be faster using array operations
        values = [list() for i in range(self.N)]
        for i, t in zip(self.neuron_index, self.spike_time):
            values[i].append(t)
        return np.array([Sequence(times) for times in values], dtype=Sequence)

    def _set_spike_time_sequences(self, spike_time_sequences, mask=None):
        if mask is not None:
            existing_times = self._get_spike_time_sequences()
            existing_times[mask] = spike_time_sequences
            spike_time_sequences = existing_times
        self._check_spike_times(spike_time_sequences)
        indices, times = self._convert_sequences_to_arrays(spike_time_sequences)
        self.set_spikes(indices, times)
    spike_time_sequences = property(fget=_get_spike_time_sequences, fset=_set_spike_time_sequences)

    def _check_spike_times(self, spike_time_sequences):
        for seq in spike_time_sequences:
            if np.any(seq.value[:-1] > seq.value[1:]):
                raise errors.InvalidParameterValueError(
                    "Spike times given to SpikeSourceArray must be in increasing order")

    def initialize(self):
        pass
