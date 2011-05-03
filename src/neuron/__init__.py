# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev: 191 $"

from pyNN.random import *
from pyNN.neuron import simulator
from pyNN import common, recording as base_recording, space, __doc__
common.simulator = simulator
base_recording.simulator = simulator

from pyNN.neuron.standardmodels.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.standardmodels.synapses import *
from pyNN.neuron.electrodes import *
from pyNN.neuron.recording import Recorder
from pyNN import standardmodels
import numpy
import logging

from neuron import h

logger = logging.getLogger("PyNN")

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.

    NEURON specific extra_params:

    use_cvode - use the NEURON cvode solver. Defaults to False.

    returns: MPI rank

    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.initializer.clear()
    simulator.state.clear()
    simulator.reset()
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if extra_params.has_key('use_cvode'):
        simulator.state.cvode.active(int(extra_params['use_cvode']))
        if extra_params.has_key('rtol'):
            simulator.state.cvode.rtol(float(extra_params['rtol']))
        if extra_params.has_key('atol'):
            simulator.state.cvode.atol(float(extra_params['atol']))
    if extra_params.has_key('default_maxstep'):
        simulator.state.default_maxstep=float(extra_params['default_maxstep'])
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    simulator.recorder_list = []
    #simulator.finalize()
        
def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()
    
reset = common.reset

initialize = common.initialize

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
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    recorder_class = Recorder
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 label=None):
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams, structure, label)
        simulator.initializer.register(self)

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create cells in NEURON.
        
        `cellclass`  -- a PyNN standard cell or a native NEURON cell class that
                       implements an as-yet-undescribed interface.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # this method should never be called more than once
        # perhaps should check for that
        assert n > 0, 'n must be a positive integer'
        celltype = cellclass(cellparams)
        cell_model = celltype.model
        cell_parameters = celltype.parameters
        self.first_id = simulator.state.gid_counter
        self.last_id = simulator.state.gid_counter + n - 1
        self.all_cells = numpy.array([id for id in range(self.first_id, self.last_id+1)], simulator.ID)
        # mask_local is used to extract those elements from arrays that apply to the cells on the current node
        self._mask_local = self.all_cells%simulator.state.num_processes==simulator.state.mpi_rank # round-robin distribution of cells between nodes
        for i,(id,is_local) in enumerate(zip(self.all_cells, self._mask_local)):
            self.all_cells[i] = simulator.ID(id)
            self.all_cells[i].parent = self
            if is_local:
                self.all_cells[i]._build_cell(cell_model, cell_parameters)
        simulator.initializer.register(*self.all_cells[self._mask_local])
        simulator.state.gid_counter += n

    def _native_rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        assert isinstance(rand_distr.rng, NativeRNG)
        rng = simulator.h.Random(rand_distr.rng.seed or 0)
        native_rand_distr = getattr(rng, rand_distr.name)
        rarr = [native_rand_distr(*rand_distr.parameters)] + [rng.repick() for i in range(self.all_cells.size-1)]
        self.tset(parametername, rarr)


PopulationView = common.PopulationView
Assembly = common.Assembly

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
        self.synapse_type = target or 'excitatory'
        
        
        ## Deal with short-term synaptic plasticity
        if self.synapse_dynamics and self.synapse_dynamics.fast:
            # need to check it is actually the Ts-M model, even though that is the only one at present!
            U = self.synapse_dynamics.fast.parameters['U']
            tau_rec = self.synapse_dynamics.fast.parameters['tau_rec']
            tau_facil = self.synapse_dynamics.fast.parameters['tau_facil']
            u0 = self.synapse_dynamics.fast.parameters['u0']
            for cell in self.post:
                cell._cell.set_Tsodyks_Markram_synapses(self.synapse_type, U, tau_rec, tau_facil, u0)
            synapse_model = 'Tsodyks-Markram'
        else:
            synapse_model = None
                
        self.connection_manager = simulator.ConnectionManager(self.synapse_type,
                                                              synapse_model=synapse_model,
                                                              parent=self)
        self.connections = self.connection_manager        
        ## Create connections
        method.connect(self)
            
        logger.info("--- Projection[%s].__init__() ---" %self.label)
               
        ## Deal with long-term synaptic plasticity
        if self.synapse_dynamics and self.synapse_dynamics.slow:
            ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
            if ddf > 0.5 and num_processes() > 1:
                # depending on delays, can run into problems with the delay from the
                # pre-synaptic neuron to the weight-adjuster mechanism being zero.
                # The best (only?) solution would be to create connections on the
                # node with the pre-synaptic neurons for ddf>0.5 and on the node
                # with the post-synaptic neuron (as is done now) for ddf<0.5
                raise NotImplementedError("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")
            stdp_parameters = self.synapse_dynamics.slow.all_parameters
            stdp_parameters['allow_update_on_post'] = int(False) # for compatibility with NEST
            long_term_plasticity_mechanism = self.synapse_dynamics.slow.possible_models
            for c in self.connections:
                c.useSTDP(long_term_plasticity_mechanism, stdp_parameters, ddf)
        
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
        logger.info("--- Projection[%s].__randomizeWeights__() ---" % self.label)
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
        logger.info("--- Projection[%s].__randomizeDelays__() ---" % self.label)
        self.setDelays(rarr)


Space = space.Space

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

record_gsyn = common.build_record('gsyn', simulator)

# ==============================================================================
