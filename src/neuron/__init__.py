# encoding: utf-8
"""
nrnpython implementation of the PyNN API.
 
$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev: 191 $"

from pyNN.random import *
from pyNN.neuron import simulator
from pyNN import common, __doc__
common.simulator = simulator

from pyNN.neuron.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.synapses import *
from pyNN.neuron.electrodes import *

import numpy
import logging

# Global variables
quit_on_end = True


# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global quit_on_end
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.dt = timestep
    simulator.reset()
    if 'quit_on_end' in extra_params:
        quit_on_end = extra_params['quit_on_end']
    if extra_params.has_key('use_cvode'):
        simulator.state.cvode.active(int(extra_params['use_cvode']))
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)
    simulator.finalize(quit_on_end)
        
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
    All cells have both an address (a tuple) and an id (an integer). If p is a
    Population object, the address and id can be inter-converted using :
    id = p[address]
    address = p.locate(id)
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
        self.recorders = {'spikes': simulator.Recorder('spikes', population=self),
                          'v': simulator.Recorder('v', population=self),
                          'gsyn': simulator.Recorder('gsyn', population=self)}
        self.label = self.label or 'population%d' % Population.nPop
        if isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
            self.celltype = cellclass(cellparams)
        else:
            self.celltype = cellclass

        # Build the arrays of cell ids
        # Cells on the local node are represented as ID objects, other cells by integers
        # All are stored in a single numpy array for easy lookup by address
        # The local cells are also stored in a list, for easy iteration
        self.all_cells, self._mask_local, self.first_id, self.last_id = simulator.create_cells(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        self.all_cells = self.all_cells.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        self.cell = self.all_cells # temporary, awaiting harmonisation
        
        simulator.initializer.register(self)
        Population.nPop += 1
        logging.info(self.describe('Creating Population "$label" of shape $dim, '+
                                   'containing `$celltype`s with indices between $first_id and $last_id'))
        logging.debug(self.describe())

    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        # Note that we generate enough random numbers for all cells on all nodes
        # but use only those relevant to this node. This ensures that the
        # sequence of random numbers does not depend on the number of nodes,
        # provided that the same rng with the same seed is used on each node.
        if isinstance(rand_distr.rng, NativeRNG):
            rng = simulator.h.Random(rand_distr.rng.seed or 0)
            native_rand_distr = getattr(rng, rand_distr.name)
            rarr = [native_rand_distr(*rand_distr.parameters)] + [rng.repick() for i in range(self.all_cells.size-1)]
        else:
            rarr = rand_distr.next(n=self.all_cells.size, mask_local=self._mask_local.flatten())
        rarr = numpy.array(rarr)
        logging.info("%s.rset('%s', %s)", self.label, parametername, rand_distr)
        for cell,val in zip(self, rarr):
            setattr(cell, parametername, val)



        
class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    nProj = 0
    
    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None,
                 synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentials
                 
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
        self.connection_manager = simulator.ConnectionManager(parent=self)
        self.connections = self.connection_manager
        
        self.synapse_type = target or 'excitatory'
        
        ## Deal with short-term synaptic plasticity
        if self.short_term_plasticity_mechanism:
            U = self._short_term_plasticity_parameters['U']
            tau_rec = self._short_term_plasticity_parameters['tau_rec']
            tau_facil = self._short_term_plasticity_parameters['tau_facil']
            u0 = self._short_term_plasticity_parameters['u0']
            for cell in self.post:
                cell._cell.use_Tsodyks_Markram_synapses(self.synapse_type, U, tau_rec, tau_facil, u0)
                
        ## Create connections
        method.connect(self)
            
        logging.info("--- Projection[%s].__init__() ---" %self.label)
               
        ## Deal with long-term synaptic plasticity
        if self.long_term_plasticity_mechanism:
            ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
            if ddf > 0.5 and num_processes() > 1:
                # depending on delays, can run into problems with the delay from the
                # pre-synaptic neuron to the weight-adjuster mechanism being zero.
                # The best (only?) solution would be to create connections on the
                # node with the pre-synaptic neurons for ddf>0.5 and on the node
                # with the post-synaptic neuron (as is done now) for ddf<0.5
                raise Exception("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")
            self._stdp_parameters['allow_update_on_post'] = int(False) # for compatibility with NEST
            for c in self.connections:
                c.useSTDP(self.long_term_plasticity_mechanism, self._stdp_parameters, ddf)
        
        # Check none of the delays are out of bounds. This should be redundant,
        # as this should already have been done in the Connector object, so
        # we could probably remove it.
        delays = [c.nc.delay for c in self.connections]
        if delays:
            assert min(delays) >= get_min_delay()
        
        Projection.nProj += 1           
    
    # --- Methods for setting connection parameters ----------------------------
    
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python
        if isinstance(rand_distr.rng, NativeRNG):
            rarr = simulator.nativeRNG_pick(len(self),
                                            rand_distr.rng,
                                            rand_distr.name,
                                            rand_distr.parameters)
        else:       
            rarr = rand_distr.next(len(self))  
        logging.info("--- Projection[%s].__randomizeWeights__() ---" % self.label)
        self.setWeights(rarr)
    
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        # If we have a native rng, we do the loops in hoc. Otherwise, we do the loops in
        # Python
        if isinstance(rand_distr.rng, NativeRNG):
            rarr = simulator.nativeRNG_pick(len(self),
                                            rand_distr.rng,
                                            rand_distr.name,
                                            rand_distr.parameters)
        else:       
            rarr = rand_distr.next(len(self))  
        logging.info("--- Projection[%s].__randomizeDelays__() ---" % self.label)
        self.setDelays(rarr)

# ==============================================================================