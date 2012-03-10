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
from pyNN import common, core, space, __doc__

from pyNN.neuron.standardmodels.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.standardmodels.synapses import *
from pyNN.neuron.standardmodels.electrodes import *
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
      Optional cvode Parameters:
      -> rtol - specify relative error tolerance
      -> atol - specify absolute error tolerance

    native_rng_baseseed - added to MPI.rank to form seed for SpikeSourcePoisson, etc.
    default_maxstep - TODO

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

    if extra_params.has_key('native_rng_baseseed'):
        simulator.state.native_rng_baseseed=int(extra_params['native_rng_baseseed'])
    else: 
        simulator.state.native_rng_baseseed=0
        
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
    
reset = common.build_reset(simulator)

initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
            num_processes, rank = common.build_state_queries(simulator)


# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _simulator = simulator
    assembly_class = Assembly
    
    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)
    

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    recorder_class = Recorder
    assembly_class = Assembly
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams, structure, initial_values, label)
        simulator.initializer.register(self)

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

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


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    _simulator = simulator
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
            self.synapse_model = 'Tsodyks-Markram'
        else:
            self.synapse_model = None
        self.connections = []
        
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
    
    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return self.connections[i]
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [self.connections[j] for j in range(*i.indices(i.stop))]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(self.connections)
    
    def _resolve_synapse_type(self):
        if self.synapse_type is None:
            self.synapse_type = weight>=0 and 'excitatory' or 'inhibitory'
        if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
            self.synapse_type += '_TM' 
    
    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(source, int) or source > simulator.state.gid_counter or source < 0:
            errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
              
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets), len(weights), len(delays))
        self._resolve_synapse_type()
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if "." in self.synapse_type: 
                    section, synapse_type = self.synapse_type.split(".") 
                    synapse_object = getattr(getattr(target._cell, section), synapse_type) 
                else: 
                    synapse_object = getattr(target._cell, self.synapse_type) 
                nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc.weight[0] = weight
                
                # if we have a mechanism (e.g. from 9ML) that includes multiple
                # synaptic channels, need to set nc.weight[1] here
                if nc.wcnt() > 1 and hasattr(target._cell, "type"):
                    nc.weight[1] = target._cell.type.synapse_types.index(self.synapse_type)
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(simulator.Connection(source, target, nc))

    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `sources`  -- a list/1D array of pre-synaptic cell IDs, or a single ID.
        `target` -- the ID of the post-synaptic cell.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(target, int) or target > simulator.state.gid_counter or target < 0:
            errmsg = "Invalid target ID: %s (gid_counter=%d)" % (target, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(sources):
            sources = [sources]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(sources) > 0
        for source in sources:
            if not isinstance(source, common.IDMixin):
                raise errors.ConnectionError("Invalid source ID: %s" % source)
              
        assert len(sources) == len(weights) == len(delays), "%s %s %s" % (len(sources),len(weights),len(delays))
                
        if target.local:
            for source, weight, delay in zip(sources, weights, delays):
                if self.synapse_type is None:
                    self.synapse_type = weight >= 0 and 'excitatory' or 'inhibitory'
                if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
                    self.synapse_type += '_TM'
                synapse_object = getattr(target._cell, self.synapse_type)  
                nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc.weight[0] = weight
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(simulator.Connection(source, target, nc))

    
    # --- Methods for setting connection parameters ----------------------------
    
    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if numpy.isscalar(value):
            for c in self:
                setattr(c, name, value)
        elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
            for c in self.connections:
                addr = (self.pre.id_to_index(c.source), self.post.id_to_index(c.target))
                try:
                    val = value[addr]
                except IndexError, e:
                    raise IndexError("%s. addr=%s" % (e, addr))
                if numpy.isnan(val):
                    raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.target))
                else:
                    setattr(c, name, val)
        elif core.is_listlike(value):
            for c,val in zip(self.connections, value):
                setattr(c, name, val)
        elif isinstance(value, RandomDistribution):
            if isinstance(value.rng, NativeRNG):
                rarr = simulator.nativeRNG_pick(len(self),
                                                value.rng,
                                                value.name,
                                                value.parameters)
            else:       
                rarr = value.next(len(self))
            for c,val in zip(self.connections, rarr):
                setattr(c, name, val)   
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")

    def get(self, parameter_name, format, gather=True):
        """
        Get the values of a given attribute (weight, delay, etc) for all
        connections on the local MPI node.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        
        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if a single such connection exists. If there are no such connections,
        X_ij will be NaN. If there are multiple such connections, the summed
        value will be given, which makes some sense for weights, but is
        pretty meaningless for delays. 
        """
        if format == 'list':
            values = [getattr(c, parameter_name) for c in self.connections]
        elif format == 'array':
            values = numpy.nan * numpy.ones((self.pre.size, self.post.size))
            for c in self.connections:
                value = getattr(c, parameter_name)
                addr = (self.pre.id_to_index(c.source), self.post.id_to_index(c.target))
                if numpy.isnan(values[addr]):
                    values[addr] = value
                else:
                    values[addr] += value
        else:
            raise Exception("format must be 'list' or 'array'")
        return values
    

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
