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
    recorder_list

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
import brian
import numpy
from itertools import izip
import scipy.sparse
from pyNN import common, errors, core

mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz

# Global variables
recorder_list = []
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
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value*units)
    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
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


class PlainNeuronGroup(brian.NeuronGroup):
    
    def __init__(self, n, equations, **kwargs):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   compile=True,
                                   clock=clock,
                                   max_delay=max_delay,
                                   freeze=True,
                                   **kwargs)
        self.parameter_names = equations._namespace.keys()
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        self.initial_values = {}
        self._S0 = self._S[:,0]

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)

class ThresholdNeuronGroup(brian.NeuronGroup):
    
    def __init__(self, n, equations, **kwargs):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   compile=True,
                                   clock=clock,
                                   max_delay=max_delay,
                                   freeze=True,
                                   **kwargs)
        self.parameter_names = equations._namespace.keys() + ['v_thresh', 'v_reset', 'tau_refrac']
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        self.initial_values = {}
        self._S0 = self._S[:,0]

    tau_refrac = _new_property('_resetfun', 'period', ms)
    v_reset    = _new_property('_resetfun', 'resetvalue', mV)
    v_thresh   = _new_property('_threshold', 'threshold', mV)

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)

class PoissonGroupWithDelays(brian.PoissonGroup):

    def __init__(self, N, rates=0):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, N, model=brian.LazyStateUpdater(),
                                   threshold=brian.PoissonThreshold(),
                                   clock=clock,
                                   max_delay=max_delay)
        self._variable_rate = True
        self.rates          = rates
        self._S0[0]         = self.rates(self.clock.t)
        self.parameter_names = ['rate', 'start', 'duration']
    
    def initialize(self):
        pass
    
            
class MultipleSpikeGeneratorGroupWithDelays(brian.MultipleSpikeGeneratorGroup):
   
    def __init__(self, spiketimes):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        thresh = brian.directcontrol.MultipleSpikeGeneratorThreshold(spiketimes)
        brian.NeuronGroup.__init__(self, len(spiketimes),
                                   model=brian.LazyStateUpdater(),
                                   threshold=thresh,
                                   clock=clock,
                                   max_delay=max_delay)
        self.parameter_names = ['spiketimes']

    def _get_spiketimes(self):
        return self._threshold.spiketimes
    def _set_spiketimes(self, spiketimes):
        assert core.is_listlike(spiketimes)
        if len(spiketimes) == 0 or numpy.isscalar(spiketimes[0]):
            spiketimes = [spiketimes for i in xrange(len(self))]
        assert len(spiketimes) == len(self), "spiketimes (length %d) must contain as many iterables as there are cells in the group (%d)." % (len(spiketimes), len(self))
        self._threshold.set_spike_times(spiketimes)
    spiketimes = property(fget=_get_spiketimes, fset=_set_spiketimes)

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
    for group in state.network.groups:
        group.initialize()    
    
# --- For implementation of access to individual neurons' parameters -----------
    
class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    gid = 0

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""    
        if n is None:
            n = gid
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        """Return a dictionary of parameters for the Brian cell model."""
        params = {}
        assert hasattr(self.parent_group, "parameter_names"), str(self.celltype)
        index = self.parent.id_to_index(self)
        for name in self.parent_group.parameter_names:
            if name in ['v_thresh', 'v_reset', 'tau_refrac']:
                # parameter shared among all cells
                params[name] = float(getattr(self.parent_group, name))
            elif name in ['rate', 'duration', 'start']:
                params[name] = getattr(self.parent_group.rates, name)[index]
            elif name == 'spiketimes':
                params[name] = getattr(self.parent_group,name)[index]
            else:
                # parameter may vary from cell to cell
                try:
                    params[name] = float(getattr(self.parent_group, name)[index])
                except TypeError, errmsg:
                    raise TypeError("%s. celltype=%s, parameter name=%s" % (errmsg, self.celltype, name))
        return params
    
    def set_native_parameters(self, parameters):
        """Set parameters of the Brian cell model from a dictionary."""
        index = self.parent.id_to_index(self)
        for name, value in parameters.items():
            if name in ['v_thresh', 'v_reset', 'tau_refrac']:
                setattr(self.parent_group, name, value)
                #logger.warning("This parameter cannot be set for individual cells within a Population. Changing the value for all cells in the Population.")
            elif name in ['rate', 'duration', 'start']:
                getattr(self.parent_group.rates, name)[index] = value
            elif name == 'spiketimes':
                all_spiketimes = [st[st>state.t] for st in self.parent_group.spiketimes]
                all_spiketimes[index] = value
                self.parent_group.spiketimes = all_spiketimes
            else:
                setattr(self.parent_group[index], name, value)
        
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

    
class SimpleCustomRefractoriness(brian.Refractoriness):
    
    @brian.check_units(period=brian.second)
    def __init__(self, resetfun, period=5*brian.msecond, state=0):
        self.period = period
        self.resetfun = resetfun
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}
        self.lastresetvalues = {}

    def __call__(self,P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period/P.clock.dt)+1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        LRV = self.lastresetvalues.get(id(P),None)
        if LRV is None:
            LRV = numpy.zeros(len(V))
            self.lastresetvalues[id(P)] = LRV
        lastspikes = P.LS.lastspikes()
        self.resetfun(P,lastspikes)             # call custom reset function 
        LRV[lastspikes] = V[lastspikes]         # store a copy of the custom resetted values
        clampedindices = P.LS[0:period] 
        V[clampedindices] = LRV[clampedindices] # clamp at custom resetted values
        
    def __repr__(self):
        return 'Custom refractory period, '+str(self.period)


class AdaptiveReset(object):

    def __init__(self, Vr= -70.6 * mV, b=0.0805 * nA):
        self.Vr = Vr
        self.b  = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr
        P.w[spikes] += self.b


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
