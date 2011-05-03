# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    create_cells()
    reset()
    run()

Classes:
    ID
    Recorder
    ConnectionManager
    Connection
    
Attributes:
    state -- a singleton instance of the _State class.
    recorder_list

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: simulator.py 926 2011-02-03 13:44:28Z apdavison $
"""

import nemo, numpy, logging, sys
from itertools import izip
from pyNN import common, errors, core, utility
from pyNN.nemo.standardmodels.cells import SpikeSourceArray, SpikeSourcePoisson

# Global variables
recorder_list     = []
spikes_array_list = []

logger = logging.getLogger("PyNN")

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self, timestep, min_delay, max_delay):
        """Initialize the simulator."""
        self.net           = nemo.Network()
        self.conf          = nemo.Configuration()
        self.initialized   = True
        self.num_processes = 1
        self.mpi_rank      = 0
        self.min_delay     = min_delay
        self.max_delay     = max_delay
        self.gid           = 0
        self.dt            = timestep
        self.simulation    = None
        self.stdp          = None
        self.verbose       = True

    def progressbar(self, N):
        self.prog = utility.ProgressBar(0, N, 20, mode='fixed')        

    @property
    def sim(self):
        if self.simulation is False:
            raise Exception("Simulation object is empty, run() needs to be called first")
        else: 
            return self.simulation
        
    @property
    def t(self):
        return self.sim.elapsed_simulation()

    def set_stdp(self, stdp):
        self.stdp = stdp
        pre   = self.stdp.timing_dependence.pre_fire(self.dt)
        post  = self.stdp.timing_dependence.pre_fire(self.dt)
        pre  *= self.stdp.weight_dependence.parameters['A_plus']
        post *= -self.stdp.weight_dependence.parameters['A_minus']        
        w_min = self.stdp.weight_dependence.parameters['w_min']
        w_max = self.stdp.weight_dependence.parameters['w_max']        
        self.conf.set_stdp_function(pre.tolist(), post.tolist(), float(w_min), float(w_max))

    def run(self, simtime):
        self.simulation = nemo.Simulation(self.net, self.conf)
        if self.verbose:
            self.progressbar(simtime)

        poissons_sources = []
        arrays_sources   = []

        for source in spikes_array_list:
            if isinstance(source.celltype, SpikeSourcePoisson):        
                poissons_sources.append(source)
            if isinstance(source.celltype, SpikeSourceArray):
                arrays_sources.append(source)
        
        for t in numpy.arange(0, simtime, self.dt):
            spikes   = []
            currents = [] 
            for source in poissons_sources:
                if source.player.do_spike(t):
                    spikes += [source]
            for source in arrays_sources:
                if source.player.next_spike == t:
                    source.player.update()                    
                    spikes += [source]
            
            #for currents in current_sources:
            #    currents.
            fired = numpy.sort(self.sim.step(spikes, currents)) 

            if self.stdp:
                self.simulation.apply_stdp(1.0)

            for recorder in recorder_list:
                if recorder.variable is "spikes":
                    recorder._add_spike(fired, self.t)
                if recorder.variable is "v":
                    recorder._add_vm(self.t)
            if self.verbose:                
                self.prog.update_amount(t)
                print self.prog, "\r",
                sys.stdout.flush()
    
    @property
    def next_id(self):        
        res = self.gid
        self.gid += 1
        return res
        

def reset():
    """Reset the state of the current network to time t = 0."""
    state.net.reset_timer()    
    
# --- For implementation of access to individual neurons' parameters -----------
    
class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        if isinstance(self.celltype, SpikeSourceArray):
            return {'spike_times' : self.player.spike_times}
        elif isinstance(self.celltype, SpikeSourcePoisson):
            return {'rate' : self.player.rate, 'duration' : self.player.duration, 
                    'start' : self.player.start}
        else:
            params = {}
            for key, value in self.celltype.indices.items():
                if state.simulation is None:
                    params[key] = state.net.get_neuron_parameter(self, value) 
                else:
                    params[key] = state.sim.get_neuron_parameter(self, value)
            return params

    def set_native_parameters(self, parameters):
        if isinstance(self.celltype, SpikeSourceArray):
            parameters['precision'] = state.dt
            self.player.reset(**parameters)
        elif isinstance(self.celltype, SpikeSourcePoisson):
            parameters['precision'] = state.dt
            self.player.reset(**parameters)    
        else:
            indices = self.celltype.indices
            for key, value in parameters.items():
                if state.simulation is None:
                    state.net.set_neuron_parameter(self, indices[key], value) 
                else:
                    state.sim.set_neuron_parameter(self, indices[key], value)
            
    def set_initial_value(self, variable, value):
        indices = self.celltype.indices.items()
        if state.simulation is None:
            state.net.set_neuron_state(self, indices[variable], value) 
        else:
            state.sim.set_neuron_state(self, indices[variable], value)
            
    def get_initial_value(self, variable):
        index = self.celltype.initial_indices[variable]
        if state.simulation is None:
            return state.net.get_neuron_state(self, index) 
        else:
            return state.sim.get_neuron_state(self, index)


class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """
    
    def __init__(self, synapse):
        """
        Create a new connection.
        
        """
        # the index is the nth non-zero element
        self.synapse = synapse 

    @property
    def target(self):
        return state.sim.get_synapse_target([self.synapse])[0]

    def _set_weight(self, w):
        pass

    def _get_weight(self):
        return state.sim.get_synapse_weight([self.synapse])[0]

    def _set_delay(self, d):
        pass

    def _get_delay(self):
        return state.sim.get_synapse_delay([self.synapse])[0]
        
    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)
    

class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_type` -- the 'physiological type' of the synapse, e.g.
                          'excitatory' or 'inhibitory',or any other key in the
                          `synapses` attibute of the celltype class.
        `synapse_model` -- not used. Present for consistency with other simulators.
        `parent` -- the parent `Projection`, if any.
        """
        self.synapse_type      = synapse_type
        self.synapse_model     = synapse_model
        self.parent            = parent
        self.sources           = []
        self.is_plastic        = False
        if self.synapse_model is "stdp_synapse":
            self.is_plastic = True
        self._connections = None
        
    def __getitem__(self, i):
        if isinstance(i, int):
            if i < len(self):
                return Connection(self.connections[i])
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [Connection(self.connections[j]) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(self.connections)
    
    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        for i in range(len(self)):
            yield self[i]
    
    @property
    def connections(self):
        if self._connections is None:
            self._connections = []
            for source in numpy.unique(self.sources):
                self._connections += list(state.net.get_synapses_from(source))
        return self._connections
    
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
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        if not isinstance(source, common.IDMixin):
            raise errors.ConnectionError("source should be an ID object, actually %s" % type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        synapse_type = self.synapse_type or "excitatory"
        if isinstance(delays, numpy.ndarray):
            delays = delays.astype(int).tolist()
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()    
        source   = int(source)
        if len(targets) == 1:
            targets = int(targets[0])
            delays  = int(delays[0])
            weights = weights[0]
        synapses = state.net.add_synapse(source, targets, delays, weights, self.is_plastic)
        self.sources.append(source)
        
    def get(self, parameter_name, format):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this manager.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        
        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if such a connection exists. If there are no such connections, X_ij will
        be NaN.
        """
        if parameter_name not in ('weight', 'delay'):
            raise Exception("Only weights and delays can be accessed by Nemo")

        if format == 'list':
            if parameter_name is "weight":
                values = list(state.sim.get_synapse_weight(self.connections))
            if parameter_name is "delay":
                values = list(state.sim.get_synapse_delay(self.connections))
        elif format == 'array':
            value_arr = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            sources  = [i.source for i in self]
            synapses = [i.synapse for i in self]
            targets  = list(state.sim.get_targets(synapses))
            addr     = self.parent.pre.id_to_index(sources), self.parent.post.id_to_index(targets)      
            if parameter_name is "weight":
                data = list(state.sim.get_weights(synapses))
            if parameter_name is "delay":
                data = list(state.sim.get_delays(synapses))          
            for idx in xrange(len(data)):
                address = addr[0][idx], addr[1][idx]
                if numpy.isnan(value_arr[address]):
                    value_arr[address] = data[idx]
                else:
                    value_arr[address] += data[idx]
            values = value_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values

    def set(self, name, value):
        """
        Set connection attributes for all connections in this manager.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        pass         

# --- Initialization, and module attributes ------------------------------------

state = None  # a Singleton, so only a single instance ever exists
#del _State
