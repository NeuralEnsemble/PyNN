# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes usable by the common implementation:

Functions:
    reset()
    run()

Classes:
    ID
    Recorder
    Connection

Attributes:
    state -- a singleton instance of the _State class.
    recorders

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
import brian
import numpy
from pyNN import common, core
from pyNN.parameters import Sequence

mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz
ampere = brian.amp

# Global variables
recorders = set([])
write_on_end = [] # a list of (population, variable, filename) combinations that should be written to file on end()
ZERO_WEIGHT = 1e-99

logger = logging.getLogger("PyNN")

# --- Internal Brian functionality --------------------------------------------

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


def nesteddictwalk(d):
    """
    Walk a nested dict structure, returning all values in all dicts.
    """
    for value1 in d.values():
        if isinstance(value1, dict):
            for value2 in nesteddictwalk(value1):  # recurse into subdict
                yield value2
        else:
            yield value1


class BaseNeuronGroup(brian.NeuronGroup):

    def __init__(self, n, equations, threshold, reset, refractory,
                 implicit=False, **parameters):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
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
                                   clock=state.simclock,
                                   max_delay=state.max_delay*ms,
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


class ThresholdNeuronGroup(BaseNeuronGroup):

    def __init__(self, n, equations, **parameters):
        threshold = brian.SimpleFunThreshold(self.check_threshold)
        reset = brian.Reset(parameters['v_reset']*mV)
        refractory = parameters['tau_refrac']*ms
        BaseNeuronGroup.__init__(self, n, equations,
                                 threshold, reset, refractory,
                                 **parameters)
        self._variable_refractory_time = True

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


# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""

    def __init__(self, timestep, min_delay, max_delay):
        """Initialize the simulator."""
        self.network       = brian.Network()
        self._set_dt(timestep)
        self.initialized   = True
        self.num_processes = 1
        self.mpi_rank      = 0
        self.min_delay     = min_delay
        self.max_delay     = max_delay
        self.gid           = 0
        self.running       = False
        self.t_start = 0

    def _get_dt(self):
        if self.network.clock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return self.network.clock.dt/ms

    def _set_dt(self, timestep):
        if self.network.clock is None or timestep != self._get_dt():
            self.network.clock = brian.Clock(dt=timestep*ms)
    dt = property(fget=_get_dt, fset=_set_dt)

    @property
    def simclock(self):
        if self.network.clock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return self.network.clock

    @property
    def t(self):
        return self.simclock.t/ms

    def run(self, simtime):
        self.running = True
        self.network.run(simtime * ms)

    def add(self, obj):
        self.network.add(obj)

    @property
    def next_id(self):
        res = self.gid
        self.gid += 1
        return res


def reset():
    """Reset the state of the current network to time t = 0."""
    state.network.reinit()
    state.running = False
    state.t_start = 0
    for group in state.network.groups:
        logger.debug("Re-initalizing %s" % group)
        group.initialize()

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    gid = 0

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        if n is None:
            n = ID.gid
        int.__init__(n)
        common.IDMixin.__init__(self)

    #def get_native_parameters(self):
    #    """Return a dictionary of parameters for the Brian cell model."""
    #    params = {}
    #    assert hasattr(self.parent_group, "parameter_names"), str(self.celltype)
    #    index = self.parent.id_to_index(self)
    #    for name in self.parent_group.parameter_names:
    #        if name in ['v_thresh', 'v_reset', 'tau_refrac']:
    #            # parameter shared among all cells
    #            params[name] = float(getattr(self.parent_group, name))
    #        elif name in ['rate', 'duration', 'start']:
    #            params[name] = getattr(self.parent_group.rates, name)[index]
    #        elif name == 'spiketimes':
    #            params[name] = getattr(self.parent_group,name)[index]
    #        else:
    #            # parameter may vary from cell to cell
    #            try:
    #                params[name] = float(getattr(self.parent_group, name)[index])
    #            except TypeError, errmsg:
    #                raise TypeError("%s. celltype=%s, parameter name=%s" % (errmsg, self.celltype, name))
    #    return params
    #
    #def set_native_parameters(self, parameters):
    #    """Set parameters of the Brian cell model from a dictionary."""
    #    index = self.parent.id_to_index(self)
    #    for name, value in parameters.items():
    #        if name in ['v_thresh', 'v_reset', 'tau_refrac']:
    #            setattr(self.parent_group, name, value)
    #            #logger.warning("This parameter cannot be set for individual cells within a Population. Changing the value for all cells in the Population.")
    #        elif name in ['rate', 'duration', 'start']:
    #            getattr(self.parent_group.rates, name)[index] = value
    #        elif name == 'spiketimes':
    #            all_spiketimes = [st[st>state.t] for st in self.parent_group.spiketimes]
    #            all_spiketimes[index] = value
    #            self.parent_group.spiketimes = all_spiketimes
    #        else:
    #            setattr(self.parent_group[index], name, value)

    def set_initial_value(self, variable, value):
        if variable is 'v':
            value *= mV
        self.parent_group.initial_values[variable][self.parent.id_to_local_index(self)] = value

    def get_initial_value(self, variable):
        return self.parent_group.initial_values[variable][self.parent.id_to_local_index(self)]


# --- For implementation of create() and Population.__init__() -----------------

class STDP(brian.STDP):
    '''
    See documentation for brian:class:`STDP` for more details. Options hidden here could be used in a more
    general manner. For example, spike interactions (all-to-all, nearest), or the axonal vs. dendritic delays
    because Brian do support those features....
    '''
    def __init__(self, C, taup, taum, Ap, Am, mu_p, mu_m, wmin=0, wmax=None,
                 delay_pre=None, delay_post=None):
        if wmax is None:
            raise AttributeError, "You must specify the maximum synaptic weight"
        wmax  = float(wmax) # removes units
        wmin  = float(wmin)
        Ap   *= wmax   # removes units
        Am   *= wmax   # removes units
        eqs = brian.Equations('''
            dA_pre/dt  = -A_pre/taup  : 1
            dA_post/dt = -A_post/taum : 1''', taup=taup, taum=taum, wmax=wmax, mu_m=mu_m, mu_p=mu_p)
        pre   = 'A_pre += Ap'
        pre  += '\nw += A_post*(w/wmax)**mu_m'

        post  = 'A_post += Am'
        post += '\nw += A_pre*(1-w/wmax)**mu_p'
        brian.STDP.__init__(self, C, eqs=eqs, pre=pre, post=post, wmin=wmin, wmax=wmax, delay_pre=None, delay_post=None, clock=None)


#class SimpleCustomRefractoriness(brian.Refractoriness):
#
#    @brian.check_units(period=brian.second)
#    def __init__(self, resetfun, period=5*brian.msecond, state=0):
#        self.period = period
#        self.resetfun = resetfun
#        self.state = state
#        self._periods = {} # a dictionary mapping group IDs to periods
#        self.statevectors = {}
#        self.lastresetvalues = {}
#
#    def __call__(self,P):
#        '''
#        Clamps state variable at reset value.
#        '''
#        # if we haven't computed the integer period for this group yet.
#        # do so now
#        if id(P) in self._periods:
#            period = self._periods[id(P)]
#        else:
#            period = int(self.period/P.clock.dt)+1
#            self._periods[id(P)] = period
#        V = self.statevectors.get(id(P),None)
#        if V is None:
#            V = P.state_(self.state)
#            self.statevectors[id(P)] = V
#        LRV = self.lastresetvalues.get(id(P),None)
#        if LRV is None:
#            LRV = numpy.zeros(len(V))
#            self.lastresetvalues[id(P)] = LRV
#        lastspikes = P.LS.lastspikes()
#        self.resetfun(P,lastspikes)             # call custom reset function
#        LRV[lastspikes] = V[lastspikes]         # store a copy of the custom resetted values
#        clampedindices = P.LS[0:period]
#        V[clampedindices] = LRV[clampedindices] # clamp at custom resetted values
#
#    def __repr__(self):
#        return 'Custom refractory period, '+str(self.period)


class AdaptiveReset(object):

    def __init__(self, Vr= -70.6 * mV, b=0.0805 * nA):
        self.Vr = Vr
        self.b  = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr
        P.w[spikes] += self.b


class IzhikevichReset(object):

    def __init__(self, Vr= -65 * mV, d=0.2 * mV/ms):
        self.Vr = Vr
        self.d  = d

    def __call__(self, P, spikes):
        P.v[spikes]  = self.Vr
        P.u[spikes] += self.d


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, brian_connection, indices, addr):
        """
        Create a new connection.

        `brian_connection` -- a Brian Connection object (may contain
                              many connections).
        `index` -- the index of the current connection within
                   `brian_connection`, i.e. the nth non-zero element
                   in the weight array.
        `indices` -- the mapping of the x, y coordinates of the established
                     connections, stored by the Projection handling those
                     connections.
        """
        # the index is the nth non-zero element
        self.bc                  = brian_connection
        self.addr                = int(indices[0]), int(indices[1])
        self.source, self.target = addr

    def _set_weight(self, w):
        w = w or ZERO_WEIGHT
        self.bc[self.addr] = w*self.bc.weight_units

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return float(self.bc[self.addr]/self.bc.weight_units)

    def _set_delay(self, d):
        self.bc.delay[self.addr] = d*ms

    def _get_delay(self):
        """Synaptic delay in ms."""
        if isinstance(self.bc, brian.DelayConnection):
            return float(self.bc.delay[self.addr]/ms)
        if isinstance(self.bc, brian.Connection):
            return float(self.bc.delay * self.bc.source.clock.dt/ms)

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)


# --- Initialization, and module attributes ------------------------------------

state = None  # a Singleton, so only a single instance ever exists
#del _State
