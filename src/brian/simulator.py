# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the Brian simulator.

Classes and attributes usable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- an instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import brian
from pyNN import common

mV = brian.mV
ms = brian.ms

# Global variables
ZERO_WEIGHT = 1e-99

logger = logging.getLogger("PyNN")


# --- For implementation of get_time_step() and similar functions --------------

class _State(common.control.BaseState):
    """Represent the simulator state."""

    def __init__(self, timestep, min_delay, max_delay):
        """Initialize the simulator."""
        super(_State, self).__init__()
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
        self.running = True
        self.network.run(simtime * ms)
        
    def run_until(self, tstop):
        self.run(tstop - self.t)

    def add(self, obj):
        self.network.add(obj)

    @property
    def next_id(self):
        res = self.gid
        self.gid += 1
        return res

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.network.reinit()
        self.running = False
        self.t_start = 0
        for group in self.network.groups:
            logger.debug("Re-initalizing %s" % group)
            group.initialize()
        assert self.t == self.t_start


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

state = None
