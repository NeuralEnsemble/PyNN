# -*- coding: utf-8 -*-
"""
Brian implementation of the PyNN API.

$Id$
"""

import logging
Set = set
#import brian_no_units_no_warnings
from pyNN.brian import simulator
from pyNN import common, recording, space, standardmodels, __doc__
common.simulator = simulator
recording.simulator = simulator
from pyNN.random import *

from pyNN.brian.cells import *
from pyNN.brian.connectors import *
from pyNN.brian.synapses import *
from pyNN.brian.electrodes import *
from pyNN.brian.recording import *

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
    return standard_cell_types

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.net = brian.Network()
    simulator.net.add(update_currents) # from electrodes
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.dt = timestep
    reset()
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()

reset = common.reset

def initialize(cells, variable, value):
    if not hasattr(cells, "__len__"):
        cells = [cells]
    parents = Set([])
    for cell in cells:
        parents.add(cell.parent_group)
    if len(parents) != 1:
        raise Exception("Initialising cells created through different create() calls at the same time not yet supported.")
    if isinstance(value, RandomDistribution):
        rarr = value.next(n=self.all_cells.size, mask_local=self._mask_local)
        value = numpy.array(rarr)
    else:
        value = value*numpy.ones((len(cells),))
    group = list(parents)[0]
    group.initial_values[variable] = value*mV
    group.initialize()

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
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 label=None):
        """
        Create a population of neurons all of the same type.
        
        size - number of cells in the Population. For backwards-compatibility, n
               may also be a tuple giving the dimensions of a grid, e.g. n=(10,10)
               is equivalent to n=100 with structure=Grid2D()
        cellclass should either be a standardized cell class (a class inheriting
        from common.standardmodels.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        structure should be a Structure instance.
        label is an optional name for the population.
        """
        common.Population.__init__(self, size, cellclass, cellparams, structure, label)
        
        self.all_cells, self._mask_local, self.first_id, self.last_id = simulator.create_cells(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        
        for id in self.local_cells:
            id.parent = self
        self.cell = self.all_cells # temporary alias, awaiting harmonization
        
        self.recorders = {'spikes': Recorder('spikes', population=self),
                          'v': Recorder('v', population=self),
                          'gsyn': Recorder('gsyn', population=self),}
        
    def initialize(self, variable, value):
        """
        Set the initial value of one of the state variables of the neurons in
        this population.
        
        `value` may either be a numeric value (all neurons set to the same
                value) or a `RandomDistribution` object (each neuron gets a
                different value)
        """
        if isinstance(value, RandomDistribution):
            rarr = value.next(n=self.all_cells.size, mask_local=self._mask_local)
            value = numpy.array(rarr)
        else:
            value = value*numpy.ones((len(self),))
        self.brian_cells.initial_values[variable] = value*mV
        self.brian_cells.initialize()
        
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
        
        self.connection_manager = simulator.ConnectionManager(self.synapse_type, parent=self)
        self.connections = self.connection_manager
        method.connect(self)


Space = space.Space
