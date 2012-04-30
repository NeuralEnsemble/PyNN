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

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import nest
from pyNN import common, errors, core
import logging
import numpy
import sys

CHECK_CONNECTIONS = False
write_on_end = [] # a list of (population, variable, filename) combinations that should be written to file on end()
recording_devices = []
recorders = set([])
populations = [] # needed for reset

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
        self.t_start = 0.0

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
        if not device._connected:
            device.connect_to_cells()
    if not state.running:
        simtime += state.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
        state.running = True
    nest.Simulate(simtime)

def reset():
    global populations
    nest.ResetNetwork()
    nest.SetKernelStatus({'time': 0.0})
    for p in populations:
        for variable, initial_value in p.initial_values.items():
            p._set_initial_value_array(variable, initial_value)
    state.running = False
    state.t_start = 0.0

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEST cell model."""
        if "source" in self.__dict__: # self is a parrot_neuron
            gid = self.source
        else:
            gid = int(self)
        return nest.GetStatus([gid])[0]

    def set_native_parameters(self, parameters):
        """Set parameters of the NEST cell model from a dictionary."""
        if hasattr(self, "source"): # self is a parrot_neuron
            gid = self.source
        else:
            gid = self
        try:
            #nest does not like numpy array if numpy was not available when it was compiled, and so we will convert them to lists whenever we encounter one
            # to avoid this inefficiency, would be better to check at import time that NEST has been built with numpy support and raise an Exception if it hasn't
            for key in parameters:
                if type(parameters[key]) == numpy.ndarray:
                   parameters[key] = parameters[key].tolist()
            nest.SetStatus([gid], [parameters])
        except: # I can't seem to catch the NESTError that is raised, hence this roundabout way of doing it.
            exc_type, exc_value, traceback = sys.exc_info()
            if exc_type == 'NESTError' and "Unsupported Numpy array type" in exc_value:
                raise errors.InvalidParameterValueError()
            else:
                raise


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, parent, index):
        """
        Create a new connection interface.

        `parent` -- a Projection instance.
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
        src.parent = self.parent.pre
        return src

    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        tgt = ID(nest.GetStatus([self.id()], 'target')[0])
        tgt.parent = self.parent.post
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
