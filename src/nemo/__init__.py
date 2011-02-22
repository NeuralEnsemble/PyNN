# -*- coding: utf-8 -*-
"""
Nemo implementation of the PyNN API.

$Id: __init__.py 927 2011-02-03 16:56:10Z pierre $
"""

import logging
Set = set
import nemo
from pyNN.nemo import simulator
from pyNN import common, recording, space, core, __doc__
common.simulator = simulator
recording.simulator = simulator
from pyNN.random import *
from pyNN.recording import files
from pyNN.nemo.standardmodels.cells import *
from pyNN.nemo.connectors import *
from pyNN.nemo.standardmodels.synapses import *
from pyNN.nemo.electrodes import *
from pyNN.nemo import electrodes
from pyNN.nemo.recording import *
from pyNN import standardmodels

logger = logging.getLogger("PyNN")

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return [obj.__name__ for obj in standard_cell_types]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=1, min_delay=1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state = simulator._State(timestep, min_delay, max_delay)
    simulator.spikes_array_list = []
    simulator.recorder_lise     = []
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    del simulator.state
    simulator.spikes_array_list = []
    simulator.recorder_list    = []
    electrodes.current_sources = []

def get_current_time():
    """Return the current time in the simulation."""
    return simulator.state.t
    
def run(simtime):    
    """Run the simulation for simtime ms."""
    simulator.state.run(simtime)
    return get_current_time()

reset      = simulator.reset
initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_time_step = common.get_time_step
get_min_delay = common.get_min_delay
get_max_delay = common.get_max_delay
num_processes = common.num_processes
rank = common.rank

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population, common.BasePopulation):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    recorder_class = Recorder

    def _create_cells(self, cellclass, cellparams=None, n=1):
        assert n > 0, 'n must be a positive integer'
        celltype = cellclass(cellparams)
        params   = celltype.parameters        
        self.all_cells  = numpy.array([simulator.ID(simulator.state.next_id) for cell in xrange(n)], simulator.ID)
        for cell in self.all_cells:
            cell.parent = self
        if isinstance(celltype, SpikeSourcePoisson):    
            simulator.spikes_array_list += self.all_cells.tolist()
            params['precision'] = simulator.state.dt
            for idx in self.all_cells:
                player = SpikeSourcePoisson.spike_player(**params)
                setattr(idx, 'player', player)
                simulator.state.net.add_neuron(int(idx), 0., 0., -80., 0, 0., -80., 0.)
        elif isinstance(celltype, SpikeSourceArray):
            ### For the moment, we model spike_source_array and spike_source_poisson
            ### as hyperpolarized neurons that are forced to fire, but this could be
            ### enhanced. A local copy of these devices is kept on the CPU, to send the
            ### spikes
            simulator.spikes_array_list += self.all_cells.tolist()
            params['precision'] = simulator.state.dt
            for idx in self.all_cells:
                player = SpikeSourceArray.spike_player(**params)
                setattr(idx, 'player', player)
                simulator.state.net.add_neuron(int(idx), 0., 0., -80., 0., -0., -80, 0)
        else:            
            ## Currently, we only have the Izhikevitch model...
            init = celltype.default_initial_values
            for idx in self.all_cells:
                simulator.state.net.add_neuron(int(idx), params['a'], params['b'], params['c'], params['d'], init['u'], init['v'], 0.)
       
        self._mask_local = numpy.ones((n,), bool) # all cells are local
        self.first_id    = self.all_cells[0]
        self.last_id     = self.all_cells[-1]

    def _set_initial_value_array(self, variable, value):
        if not hasattr(value, "__len__"):
            value = value*numpy.ones((len(self),))        
       
PopulationView = common.PopulationView
Assembly = common.Assembly

class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentialss
                 
        target - string specifying which synapse on the postsynaptic cell to
                 connect to
                 
        If source and/or target are not given, default values are used.
        
        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - specify an RNG object to be used by the Connector.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method,
                                   source, target, synapse_dynamics, label, rng)
        
        self._method           = method
        self._connections      = None
        self.synapse_type      = target or 'excitatory'
        
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                if self.synapse_dynamics.slow:
                    raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
                else:
                    raise Exception("Tsodyks Markram synapses not implemented in Nemo yet")
            elif self.synapse_dynamics.slow:
                self._plasticity_model = "stdp_synapse"
                if simulator.state.stdp is None:
                    simulator.state.set_stdp(self.synapse_dynamics.slow)
                if self.synapse_dynamics.slow == simulator.state.stdp:
                    pass                     
                else:
                    raise Exception("Only one STDP model can be handle by Nemo. Two detected !") 
        else:        
            self._plasticity_model = "static_synapse"
                                        
        self.connection_manager = simulator.ConnectionManager(self.synapse_type, self._plasticity_model, parent=self)
        self.connections = self.connection_manager        
        method.connect(self)

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        lines = numpy.empty((len(self.connection_manager), 4))        
        lines[:,0] = [i.source for i in self.connection_manager]
        lines[:,1] = [i.target for i in self.connection_manager]            
        if compatible_output:
            lines[:,0] = self.pre.id_to_index(lines[:, 0]) 
            lines[:,1] = self.post.id_to_index(lines[:, 1])    
        synapses   = [i.synapse for i in self.connection_manager]  
        lines[:,2] = list(simulator.state.sim.get_weights(synapses))
        lines[:,3] = list(simulator.state.sim.get_delays(synapses))         
        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
        file.close()                            

Space = space.Space

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

record      = common.build_record('spikes', simulator)
record_v    = common.build_record('v', simulator)
record_gsyn = common.build_record('gsyn', simulator)


# ==============================================================================
