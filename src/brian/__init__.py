# -*- coding: utf-8 -*-
"""
Brian implementation of the PyNN API.

$Id$
"""

import logging
Set = set
#import brian_no_units_no_warnings
from pyNN.brian import simulator
from pyNN import common, recording, space, standardmodels, core, __doc__
common.simulator = simulator
recording.simulator = simulator
from pyNN.random import *
from pyNN.recording import files
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
    brian.set_global_preferences(**extra_params)
    simulator.state = simulator._State(timestep, min_delay, max_delay)
    simulator.state.add(update_currents) # from electrodes
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.dt        = timestep
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    simulator.recorder_list = []
    electrodes.current_sources = []
    for item in simulator.state.network.groups + simulator.state.network._all_operations:
        del item    
    del simulator.state

def get_current_time():
    """Return the current time in the simulation."""
    return simulator.state.t
    
def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.state.run(simtime)
    return get_current_time()

reset = simulator.reset

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
        """
        Create cells in Brian.
        
        `cellclass`  -- a PyNN standard cell or a native Brian cell class.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # currently, we create a single NeuronGroup for create(), but
        # arguably we should use n NeuronGroups each containing a single cell
        # either that or use the subgroup() method in connect(), etc
        assert n > 0, 'n must be a positive integer'
        if isinstance(cellclass, basestring):  # celltype is not a standard cell
            try:
                eqs = brian.Equations(cellclass)
            except Exception, errmsg:
                raise errors.InvalidModelError(errmsg)
            v_thresh   = cellparams['v_thresh'] * mV
            v_reset    = cellparams['v_reset'] * mV
            tau_refrac = cellparams['tau_refrac'] * ms
            brian_cells = brian.NeuronGroup(n,
                                            model=eqs,
                                            threshold=v_thresh,
                                            reset=v_reset,
                                            clock=state.simclock,
                                            compile=True,
                                            max_delay=state.max_delay)
            cell_parameters = cellparams or {}
        elif isinstance(cellclass, type) and issubclass(cellclass, standardmodels.StandardCellType):
            celltype = cellclass(cellparams)
            cell_parameters = celltype.parameters
            if isinstance(celltype, cells.SpikeSourcePoisson):    
                fct = celltype.rates(cell_parameters['start'], cell_parameters['duration'], cell_parameters['rate'], n)
                brian_cells = simulator.PoissonGroupWithDelays(n, rates=fct)
            elif isinstance(celltype, cells.SpikeSourceArray):
                spike_times = cell_parameters['spiketimes']
                brian_cells = simulator.MultipleSpikeGeneratorGroupWithDelays([spike_times for i in xrange(n)])
            else:
                params = {'threshold'  : celltype.threshold, 
                          'reset'      : celltype.reset}
                if cell_parameters.has_key('tau_refrac'):                 
                    params['refractory'] = cell_parameters['tau_refrac'] * ms
                if hasattr(celltype, 'extra'):
                    params.update(celltype.extra)
                brian_cells = simulator.ThresholdNeuronGroup(n, cellclass.eqs, **params)
                
        elif isinstance(cellclass, type) and issubclass(cellclass, standardmodels.ModelNotAvailable):
            raise NotImplementedError("The %s model is not available for this simulator." % cellclass.__name__)
        else:
            raise Exception("Invalid cell type: %s" % type(cellclass))    
    
        if cell_parameters:
            for key, value in cell_parameters.items():
                setattr(brian_cells, key, value)
        # should we globally track the IDs used, so as to ensure each cell gets a unique integer? (need only track the max ID)
        self.all_cells = numpy.array([simulator.ID(cell) for cell in xrange(len(brian_cells))], simulator.ID)
        for cell in self.all_cells:
            cell.parent = self
            cell.parent_group = brian_cells
       
        self._mask_local = numpy.ones((n,), bool) # all cells are local. This doesn't seem very efficient.
        self.first_id    = self.all_cells[0]
        self.last_id     = self.all_cells[-1]
        self.brian_cells = brian_cells
        simulator.state.network.add(brian_cells)

    def initialize(self, variable, value):
        """
        Set the initial value of one of the state variables of the neurons in
        this population.
        
        `value` may either be a numeric value (all neurons set to the same
                value) or a `RandomDistribution` object (each neuron gets a
                different value)
        """
        if isinstance(value, RandomDistribution):
            rarr  = value.next(n=self.all_cells.size, mask_local=self._mask_local)
            value = numpy.array(rarr)
        else:
            value = value*numpy.ones((len(self),))
        if variable is 'v':
            self.brian_cells.initial_values[variable] = value*mV
        self.brian_cells.initialize()
        self.initial_values[variable] = core.LazyArray(self.size, value)


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
        
        if isinstance(presynaptic_population, common.Assembly) or isinstance(postsynaptic_population, common.Assembly):
            raise Exception("Projections with Assembly objects are not working yet in Brian")
        
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                if self.synapse_dynamics.slow:
                    raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
                else:
                    self._plasticity_model = "tsodyks_markram_synapse"
            elif synapse_dynamics.slow:
                self._plasticity_model = "stdp_synapse"
        else:        
            self._plasticity_model = "static_synapse"
                                        
        self.connection_manager = simulator.ConnectionManager(self.synapse_type, self._plasticity_model, parent=self)
        self.connections = self.connection_manager        
        method.connect(self)
        self.connection_manager._finalize()
        if self._plasticity_model != "static_synapse":
            synapses = self.connections.brian_connections
            if self._plasticity_model is "stdp_synapse": 
                parameters   = self.synapse_dynamics.slow.all_parameters
                if common.is_conductance(self.post[0]):
                    units = uS
                else:
                    units = nA
                stdp = simulator.STDP(synapses, 
                                      parameters['tau_plus'] * ms,
                                      parameters['tau_minus'] * ms,
                                      parameters['A_plus'],
                                      -parameters['A_minus'],
                                      parameters['mu_plus'],
                                      parameters['mu_minus'],
                                      wmin = parameters['w_min'] * units,
                                      wmax = parameters['w_max'] * units)
                simulator.state.add(stdp)
            elif self._plasticity_model is "tsodyks_markram_synapse":
                parameters   = self.synapse_dynamics.fast.parameters
                stp = brian.STP(synapses, parameters['tau_rec'] * ms, 
                                          parameters['tau_facil'] * ms, 
                                          parameters['U'])
                simulator.state.add(stp)
                
    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        import operator
        lines = numpy.empty((len(self.connection_manager), 4))
        bc    = self.connection_manager.brian_connections    
        lines[:,0], lines[:,1] = self.connection_manager.indices
        lines[:,2] = bc.W.alldata / bc.weight_units
        if isinstance(bc, brian.DelayConnection):
            lines[:,3] = bc.delay.alldata / ms
        else:
            lines[:,3] = bc.delay * bc.source.clock.dt / ms
        
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
        file.close()

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
