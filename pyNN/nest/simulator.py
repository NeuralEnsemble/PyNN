# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the NEST simulator.

Classes and attributes usable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- a singleton instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import nest
import logging
import tempfile
import numpy
from pyNN import common, random
from pyNN.core import reraise

logger = logging.getLogger("PyNN")
name = "NEST"  # for use in annotating output data

# --- For implementation of get_time_step() and similar functions --------------

def nest_property(name, dtype):
    """Return a property that accesses a NEST kernel parameter"""
    def _get(self):
        return nest.GetKernelStatus(name)
    def _set(self, val):
        try:
            nest.SetKernelStatus({name: dtype(val)})
        except nest.NESTError as e:
            reraise(e, "%s = %s (%s)" % (name, val, type(val)))
    return property(fget=_get, fset=_set)


class _State(common.control.BaseState):
    """Represent the simulator state."""

    def __init__(self):
        super(_State, self).__init__()
        self.initialized = False
        self.optimize = False
        self.spike_precision = "off_grid"
        self.verbosity = "warning"
        self._cache_num_processes = nest.GetKernelStatus()['num_processes'] # avoids blocking if only some nodes call num_processes
                                                                            # do the same for rank?
        # allow NEST to erase previously written files (defaut with all the other simulators)
        nest.SetKernelStatus({'overwrite_files' : True})
        self.tempdirs = []
        self.recording_devices = []
        self.populations = [] # needed for reset

    @property
    def t(self):
        return max(nest.GetKernelStatus('time') - self.dt, 0.0)  # note that we always simulate one time step past the requested time

    dt = nest_property('resolution', float)

    threads = nest_property('local_num_threads', int)

    grng_seed = nest_property('grng_seed', int)

    rng_seeds = nest_property('rng_seeds', list)

    @property
    def min_delay(self):
        # this rather complex implementation is needed to handle min_delay='auto'
        kernel_delay = nest.GetKernelStatus('min_delay')
        syn_delay = nest.GetDefaults('static_synapse')['min_delay']
        if syn_delay == numpy.inf or syn_delay > 1e300:
            return kernel_delay
        else:
            return max(kernel_delay, syn_delay)

    def set_delays(self, min_delay, max_delay):
        if min_delay != 'auto': 
            min_delay = float(min_delay)
            max_delay = float(max_delay)
            for synapse_model in nest.Models(mtype='synapses'):
                nest.SetDefaults(synapse_model, {'delay': min_delay,
                                                 'min_delay': min_delay,
                                                 'max_delay': max_delay})

    @property
    def max_delay(self):
        return nest.GetDefaults('static_synapse')['max_delay']

    @property
    def num_processes(self):
        return self._cache_num_processes

    @property
    def mpi_rank(self):
        return nest.Rank()

    def _get_spike_precision(self):
        ogs = nest.GetKernelStatus('off_grid_spiking')
        return ogs and "off_grid" or "on_grid"
    def _set_spike_precision(self, precision):
        if precision == 'off_grid':
            nest.SetKernelStatus({'off_grid_spiking': True})
            self.default_recording_precision = 15
        elif precision == 'on_grid':
            nest.SetKernelStatus({'off_grid_spiking': False})
            self.default_recording_precision = 3
        else:
            raise ValueError("spike_precision must be 'on_grid' or 'off_grid'")
    spike_precision = property(fget=_get_spike_precision, fset=_set_spike_precision)

    def _set_verbosity(self, verbosity):
        nest.sli_run("M_%s setverbosity" % verbosity.upper())
    verbosity = property(fset=_set_verbosity)

    def run(self, simtime):
        """Advance the simulation for a certain time."""
        for population in self.populations:
            if population._deferred_parrot_connections:
                population._connect_parrot_neurons()
        for device in self.recording_devices:
            if not device._connected:
                device.connect_to_cells()
                device._local_files_merged = False
        if not self.running and simtime > 0:
            simtime += self.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
            self.running = True
        nest.Simulate(simtime)

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def reset(self):
        nest.ResetNetwork()
        nest.SetKernelStatus({'time': 0.0})
        for p in self.populations:
            for variable, initial_value in p.initial_values.items():
                p._set_initial_value_array(variable, initial_value)
        self.running = False
        self.t_start = 0.0
        self.segment_counter += 1

    def clear(self):
        self.populations = []
        self.recording_devices = []
        self.recorders = set()
        # clear the sli stack, if this is not done --> memory leak cause the stack increases
        nest.sr('clear')
        # reset the simulation kernel
        nest.ResetKernel()
        # set tempdir
        tempdir = tempfile.mkdtemp()
        self.tempdirs.append(tempdir) # append tempdir to tempdirs list
        nest.SetKernelStatus({'data_path': tempdir,})
        self.segment_counter = -1
        self.reset()


# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


# --- For implementation of connect() and Connector classes --------------------

class Connection(common.Connection):
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
        return self.parent.nest_connections[self.index]

    @property
    def source(self):
        """The ID of the pre-synaptic neuron."""
        src = ID(nest.GetStatus([self.id()], 'source')[0])
        src.parent = self.parent.pre
        return src
    presynaptic_cell = source

    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        tgt = ID(nest.GetStatus([self.id()], 'target')[0])
        tgt.parent = self.parent.post
        return tgt
    postsynaptic_cell = target

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
