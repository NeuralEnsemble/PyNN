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
        self.time          = 0
        self._fired        = []

    @property
    def sim(self):
        if self.simulation is None:
            raise Exception("Simulation object is empty, run() needs to be called first")
        else: 
            return self.simulation
        
    @property
    def t(self):
        return self.time

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
        
        if self.simulation is None:
            self.simulation = nemo.Simulation(self.net, self.conf)

        arrays_sources   = []

        for source in spikes_array_list:
            if isinstance(source.celltype, SpikeSourceArray):
                arrays_sources.append(source)
        
        for t in numpy.arange(self.time, self.time+simtime, self.dt):
            spikes   = []
            currents = []
            for source in arrays_sources:
                if source.player.next_spike == t:
                    source.player.update()                    
                    spikes += [source]
            for recorder in recorder_list:
                if recorder.variable is "spikes":
                    recorder._add_spike(self._fired, self.t)
                if recorder.variable is "v":
                    recorder._add_vm(self.t)
                if recorder.variable is "gsyn":
                    recorder._add_gsyn(self.t)

            
            self._fired = self.sim.step(spikes, currents)
            self.time  += self.dt
        
            if self.stdp:
                self.simulation.apply_stdp(self.dt)
               
    @property
    def next_id(self):        
        res = self.gid
        self.gid += 1
        return res
        

def reset():
    """Reset the state of the current network to time t = 0."""
    state.time   = 0
    state._fired = []
    
# --- For implementation of access to individual neurons' parameters -----------
    
class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        if isinstance(self.celltype, SpikeSourceArray):
            return {'spike_times' : self.player.spike_times}
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
        else:
            indices = self.celltype.indices
            for key, value in parameters.items():
                if state.simulation is None:
                    state.net.set_neuron_parameter(self, indices[key], value) 
                else:
                    state.sim.set_neuron_parameter(self, indices[key], value)
            
    def set_initial_value(self, variable, value):
        indices = self.celltype.initial_indices
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

    @property
    def source(self):
        return state.sim.get_synapse_source([self.synapse])[0]

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
       

# --- Initialization, and module attributes ------------------------------------

state = None  # a Singleton, so only a single instance ever exists
#del _State
