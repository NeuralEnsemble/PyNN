# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the MOOSE simulator.

Functions and classes useable by the common implementation:
    run()


All other functions and classes are private, and should not be used by other
modules.
    
$Id:$
"""

import moose
from pyNN import common, core

# global variables
recorder_list = []

ms = 1e-3
in_ms = 1.0/ms

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.ctx = moose.PyMooseBase.getContext()
        self.gid_counter = 0
        self.num_processes = 1 # we're not supporting MPI
        self.mpi_rank = 0      # for now
        self.min_delay = 0.0
        self.max_delay = 1e12

    @property
    def t(self):
        return self.ctx.getCurrentTime()*in_ms
    
    def __get_dt(self):
        return self.ctx.getClocks()[0]*in_ms
    def __set_dt(self, dt):
        print "setting dt to %g ms" % dt
        self.ctx.setClock(0, dt*ms, 0) # integration clock
        self.ctx.setClock(1, dt*ms, 1) # ?
        self.ctx.setClock(2, dt*ms, 0) # recording clock
    dt = property(fget=__get_dt, fset=__set_dt)

def run(simtime):
    print "simulating for %g ms" % simtime
    state.ctx.reset()
    state.ctx.step(simtime*ms)


# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__
    
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def _build_cell(self, cell_model, cell_parameters):
        """
        Create a cell in MOOSE, and register its global ID.
        
        `cell_model` -- one of the cell classes defined in the
                        `moose.cells` module (more generally, any class that
                        implements a certain interface, but I haven't
                        explicitly described that yet).
        `cell_parameters` -- a dictionary containing the parameters used to
                             initialise the cell model.
        """
        id = int(self)
        self._cell = cell_model("neuron%d" % id, **cell_parameters)          # create the cell object
        
    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEURON cell model."""
        D = {}
        for name in self._cell.parameter_names:
            D[name] = getattr(self._cell, name)
        return D
    
    def set_native_parameters(self, parameters):
        """Set parameters of the NEURON cell model from a dictionary."""
        for name, val in parameters.items():
            setattr(self._cell, name, val)


class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_model` -- either None or 'Tsodyks-Markram'.
        `parent` -- the parent `Projection`
        """
        assert parent is not None
        self.connections = []
        self.parent = parent
        self.synapse_type = synapse_type
        self.synapse_model = synapse_model
    
    def connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(source, int) or source > state.gid_counter or source < 0:
            errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]
            
        weights = weights*1000.0 # scale units
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        # need to scale weights for appropriate units
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if not isinstance(target, common.IDMixin):
                    raise errors.ConnectionError("Invalid target ID: %s" % target)
                if self.synapse_type == "excitatory":
                    synapse_object = target._cell.synE
                elif self.synapse_type == "inhibitory":
                    synapse_object = target._cell.synI
                else:
                    synapse_object = getattr(target._cell, self.synapse_type)
                source._cell.source.connect('event', synapse_object, 'synapse')
                print "setting weight", weight
                synapse_object.setWeight(0, weight) # presumably the first arg is the connection number?
                synapse_object.setDelay(0, delay)

state = _State()  # a Singleton, so only a single instance ever exists
del _State