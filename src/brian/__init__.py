# -*- coding: utf-8 -*-
"""
Brian implementation of the PyNN API.

$Id$
"""

import logging
#import brian_no_units_no_warnings
from pyNN.brian import simulator
from pyNN import common, recording, __doc__
common.simulator = simulator
recording.simulator = simulator

from pyNN.brian.cells import *
from pyNN.brian.connectors import *
from pyNN.brian.synapses import *
from pyNN.brian.electrodes import *
from pyNN.brian.recording import *

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return standard_cell_types

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.dt = timestep
    simulator.reset()
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()


# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time = common.get_current_time
get_time_step = common.get_time_step
get_min_delay = common.get_min_delay
get_max_delay = common.get_max_delay
num_processes = common.num_processes
rank = common.rank

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.create
connect = common.connect
set = common.set
record = common.build_record('spikes', simulator)
record_v = common.build_record('v', simulator)
record_gsyn = common.build_record('gsyn', simulator)

# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    nPop = 0
    
    def __init__(self, dims, cellclass, cellparams=None, label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
          integer, for a one-dimensional population.
          e.g., (10,10) will create a two-dimensional population of size 10x10.
        cellclass should either be a standardized cell class (a class inheriting
        from common.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        label is an optional name for the population.
        """
        common.Population.__init__(self, dims, cellclass, cellparams, label)
        
        if isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
            self.celltype = cellclass(cellparams)
        
        self.all_cells, self._mask_local, self.first_id, self.last_id = simulator.create_cells(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        self.all_cells = self.all_cells.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        
        for id in self.local_cells:
            id.parent = self
        self.cell = self.all_cells # temporary alias, awaiting harmonization
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {'spikes': Recorder('spikes', population=self),
                          'v': Recorder('v', population=self),
                          'gsyn': Recorder('gsyn', population=self),}
        Population.nPop += 1
        
    def meanSpikeCount(self, gather=True):
        """ 
        Returns the mean number of spikes per neuron. 
        """
        rec = self.recorders['spikes']
        return float(rec._devices[0].nspikes)/len(rec.recorded) 


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
        
        self._method  = method
        self._connections = None
        self._plasticity_model = "static_synapse"
        self.synapse_type = target
        
        self.connection_manager = simulator.ConnectionManager(parent=self)
        self.connections = self.connection_manager
        method.connect(self)

        