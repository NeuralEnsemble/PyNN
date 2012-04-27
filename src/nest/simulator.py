# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes usable by the common implementation:

Functions:
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


:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import nest
from pyNN import common
import logging

CHECK_CONNECTIONS = False
recorder_list = []
recording_devices = []

global net
net    = None
logger = logging.getLogger("PyNN")

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""

    def __init__(self):

        self.initialized = False
        self.running     = False
        self.optimize    = False
        self.spike_precision = "on_grid"
        self.default_recording_precision = 3
        self._cache_num_processes = nest.GetKernelStatus()['num_processes'] # avoids blocking if only some nodes call num_processes
                                                                            # do the same for rank?

    @property
    def t(self):
        return nest.GetKernelStatus()['time']

    dt = property(fget=lambda self: nest.GetKernelStatus()['resolution'],
                  fset=lambda self, timestep: nest.SetKernelStatus({'resolution': timestep}))

    @property
    def min_delay(self):
        return nest.GetDefaults('static_synapse')['min_delay']

    @property
    def max_delay(self):
        # any reason why not nest.GetKernelStatus()['min_delay']?
        return nest.GetDefaults('static_synapse')['max_delay']

    @property
    def num_processes(self):
        return self._cache_num_processes

    @property
    def mpi_rank(self):
        return nest.Rank()

    @property
    def num_threads(self):
        return nest.GetKernelStatus()['local_num_threads']


def run(simtime):
    """Advance the simulation for a certain time."""
    for device in recording_devices:
        device.connect_to_cells()
    if not state.running:
        simtime += state.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
        state.running = True
    nest.Simulate(simtime)

def reset():
    nest.ResetNetwork()
    nest.SetKernelStatus({'time': 0.0})
    state.running = False

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, parent, index):
        """
        Create a new connection interface.

        `parent` -- a ConnectionManager instance.
        `index` -- the index of this connection in the parent.
        """
        self.parent = parent
        self.index = index

    def id(self):
        """Return a tuple of arguments for `nest.GetConnection()`.
        """
        return self.parent.connections[self.index]

    @property
    def source(self):
        """The ID of the pre-synaptic neuron."""
        src = ID(nest.GetStatus([self.id()], 'source')[0])
        src.parent = self.parent.parent.pre
        return src

    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        tgt = ID(nest.GetStatus([self.id()], 'target')[0])
        tgt.parent = self.parent.parent.post
        return tgt

    def _set_weight(self, w):
        nest.SetStatus([self.id()], 'weight', w*1000.0)

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        w_nA = nest.GetStatus([self.id()], 'weight')[0]
        if self.parent.synapse_type == 'inhibitory' and common.is_conductance(self.target):
            w_nA *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
        return 0.001*w_nA

    def _set_delay(self, d):
        nest.SetStatus([self.id()], 'delay', d)

    def _get_delay(self):
        """Synaptic delay in ms."""
        return nest.GetStatus([self.id()], 'delay')[0]

    weight = property(_get_weight, _set_weight)
    delay  = property(_get_delay, _set_delay)

def generate_synapse_property(name):
    def _get(self):
        return nest.GetStatus([self.id()], name)[0]
    def _set(self, val):
        nest.SetStatus([self.id()], name, val)
    return property(_get, _set)
setattr(Connection, 'U', generate_synapse_property('U'))
setattr(Connection, 'tau_rec', generate_synapse_property('tau_rec'))
setattr(Connection, 'tau_facil', generate_synapse_property('tau_fac'))
setattr(Connection, 'u0', generate_synapse_property('u0'))
setattr(Connection, '_tau_psc', generate_synapse_property('tau_psc'))


# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State
