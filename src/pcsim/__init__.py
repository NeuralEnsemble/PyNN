# encoding: utf-8
"""
pypcsim implementation of the PyNN API. 

    Dejan Pecevski   dejan@igi.tugraz.at
    Thomas Natschlaeger   thomas.natschlaeger@scch.at
        
    December 2006
    $Id$
"""
__version__ = "$Revision$"

import sys

import pyNN.random
from pyNN import common
from pyNN.pcsim import simulator
common.simulator = simulator
import os.path
import types
import sys
import numpy
import pypcsim
from pyNN.pcsim.cells import *
from pyNN.pcsim.connectors import *
from pyNN.pcsim.synapses import *
from pyNN.pcsim.electrodes import *

try:
    import tables
except ImportError:
    pass
import exceptions
from datetime import datetime
import operator


Set = set
ID = simulator.ID

# ==============================================================================
#   Utility classes
# ==============================================================================

# Implementation of the NativeRNG
class NativeRNG(pyNN.random.NativeRNG):
    def __init__(self, seed=None, type='MersenneTwister19937'):
        pyNN.random.AbstractRNG.__init__(self, seed)
        self.rndEngine = getattr(pypcsim, type)()
        if not self.seed:
            self.seed = int(datetime.today().microsecond)
        self.rndEngine.seed(self.seed)
    
    def next(self, n=1, distribution='Uniform', parameters={'a':0,'b':1}, mask_local=None):        
        """Return n random numbers from the distribution.
        If n is 1, return a float, if n > 1, return a numpy array,
        if n <= 0, raise an Exception."""
        distribution_type = getattr(pypcsim, distribution + "Distribution")
        if isinstance(parameters, dict):
            dist = apply(distribution_type, (), parameters)
        else:
            dist = apply(distribution_type, tuple(parameters), {})
        values = [ dist.get(self.rndEngine) for i in xrange(n) ]
        if n == 1:
            return values[0]
        else:
            return values 
        
        


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

class WDManager(object):
    
    def getWeight(self, w=None):
        if w is not None:
            weight = w
        else:
            weight = 1.
        return weight
        
    def getDelay(self, d=None):
        if d is not None:
            delay = d
        else:
            delay = simulator.state.min_delay
        return delay
    
    def convertWeight(self, w, conductance):
        if conductance:
            w_factor = 1e-6 # Convert from µS to S
        else:
            w_factor = 1e-9 # Convert from nA to A
        if isinstance(w, pyNN.random.RandomDistribution):
            weight = pyNN.random.RandomDistribution(w.name, w.parameters, w.rng)
            if weight.name == "uniform":
                (w_min, w_max) = weight.parameters
                weight.parameters = (w_factor*w_min, w_factor*w_max)
            elif weight.name ==  "normal":
                (w_mean, w_std) = weight.parameters
                weight.parameters = (w_factor*w_mean, w_factor*w_std)
            else:
                print "WARNING: no conversion of the weights for this particular distribution"
        else:
            weight = w*w_factor
        return weight
    
    def reverse_convertWeight(self, w, conductance):
        if conductance:
            w_factor = 1e6 # Convert from S to µS
        else:
            w_factor = 1e9 # Convert from A to nA
        return w*w_factor
     
    def convertDelay(self, d):
        
        if isinstance(d, pyNN.random.RandomDistribution):
            delay = pyNN.random.RandomDistribution(d.name, d.parameters, d.rng)
            if delay.name == "uniform":
                (d_min, d_max) = delay.parameters
                delay.parameters = (d_min/1000., d_max/1000.)
            elif delay.name ==  "normal":
                (d_mean, d_std) = delay.parameters
                delay.parameters = (d_mean/1000., w_std)
        else:
            delay = d/1000.
        return delay
        
 
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    For pcsim, the possible arguments are 'construct_rng_seed' and 'simulation_rng_seed'.
    """
 
    
    simulator.state.t = 0
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if simulator.state.constructRNGSeed is None:
        if extra_params.has_key('construct_rng_seed'):
            construct_rng_seed = extra_params['construct_rng_seed']
        else:
            construct_rng_seed = datetime.today().microsecond
        simulator.state.constructRNGSeed = construct_rng_seed
    if simulator.state.simulationRNGSeed is None:
        if extra_params.has_key('simulation_rng_seed'):
            simulation_rng_seed = extra_params['simulation_rng_seed']
        else:
            simulation_rng_seed = datetime.today().microsecond
        simulator.state.simulationRNGSeed = simulation_rng_seed
    if extra_params.has_key('threads'):
        simulator.net = pypcsim.DistributedMultiThreadNetwork(extra_params['threads'],
                                                                  pypcsim.SimParameter( pypcsim.Time.ms(timestep), pypcsim.Time.ms(min_delay), pypcsim.Time.ms(max_delay), simulator.state.constructRNGSeed, simulator.state.simulationRNGSeed))
    else:
        simulator.net = pypcsim.DistributedSingleThreadNetwork(pypcsim.SimParameter( pypcsim.Time.ms(timestep), pypcsim.Time.ms(min_delay), pypcsim.Time.ms(max_delay), simulator.state.constructRNGSeed, simulator.state.simulationRNGSeed))
    simulator.spikes_multi_rec = {}
    simulator.vm_multi_rec = {}
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    return simulator.net.mpi_rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    
    for filename, rec in simulator.vm_multi_rec.items():
        rec.saveValuesText(compatible_output=compatible_output)
    for filename, rec in simulator.spikes_multi_rec.items():
        rec.saveSpikesText(compatible_output=compatible_output)    
    simulator.vm_multi_rec = {}     
    simulator.spikes_multi_rec = {}
     

def run(simtime):
    """Run the simulation for simtime ms."""
    
    simulator.state.t += simtime
    simulator.net.advance(int(simtime / simulator.state.dt ))
    return get_current_time()

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

#def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
#    """Connect a source of spikes to a synaptic target. source and target can
#    both be individual cells or lists of cells, in which case all possible
#    connections are made with probability p, using either the random number
#    generator supplied, or the default rng otherwise.
#    Weights should be in nA or µS."""
#    
#    if weight is None:  weight = 0.0
#    if delay  is None:  delay = simulator.state.min_delay
#    if delay < get_min_delay():
#        raise common.ConnectionError("Delay (%g ms) must be >= the minimum delay (%g ms)" % (delay, get_min_delay()))
#    # Convert units
#    delay = delay / 1000 # Delays in pcsim are specified in seconds
#    if isinstance(target, list):
#        firsttarget = target[0]
#    else:
#        firsttarget = target
#    try:
#        if hasattr(simulator.net.object(firsttarget),'ErevExc'):
#            weight = 1e-6 * weight # Convert from µS to S    
#        else:
#            weight = 1e-9 * weight # Convert from nA to A
#    except exceptions.Exception, e: # non-existent connection
#        raise common.ConnectionError(e)
#    # Create synapse factory
#    syn_factory = 0
#    if synapse_type is None:
#        if weight >= 0:  # decide whether to connect to the excitatory or inhibitory response 
#            syn_target_id = 1
#        else:
#            syn_target_id = 2
#        syn_factory = pypcsim.SimpleScalingSpikingSynapse(syn_target_id, weight, delay)
#    else:
#        if isinstance(synapse_type, type):
#            syn_factory = synapse_type
#        elif isinstance(synapse_type, str):
#            if synapse_type == 'excitatory':
#                syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, weight, delay)
#            elif synapse_type == 'inhibitory':
#                syn_factory = pypcsim.SimpleScalingSpikingSynapse(2, weight, delay)
#            else:
#                eval('syn_factory = ' + synapse_type + '()')
#            syn_factory.W = weight;
#            syn_factory.delay = delay;
#    # Create connections
#    try:
#        if type(source) != types.ListType and type(target) != types.ListType:
#            connections = simulator.net.connect(source, target, syn_factory)
#            if not common.is_listlike(connections):
#                connections = [connections]
#            return connections
#        else:
#            if type(source) != types.ListType:
#                source = [source]
#            if type(target) != types.ListType:
#                target = [target]
#            src_popul = pypcsim.SimObjectPopulation(simulator.net, source)
#            dest_popul = pypcsim.SimObjectPopulation(simulator.net, target)
#            connections = pypcsim.ConnectionsProjection(src_popul, dest_popul, syn_factory, pypcsim.RandomConnections(p), pypcsim.SimpleAllToAllWiringMethod(simulator.net), True)
#            return connections.idVector()
#    except exceptions.TypeError, e:
#        raise common.ConnectionError(e)
#    except exceptions.Exception, e:
#        raise common.ConnectionError(e)
connect = common.connect
set = common.set    
record = common.build_record('spikes', simulator)
record_v = common.build_record('v', simulator)
            

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
        ##if isinstance(dims, int): # also allow a single integer, for a 1D population
        ##    #print "Converting integer dims to tuple"
        ##    dims = (dims,)
        ##elif len(dims) > 3:
        ##    raise exceptions.AttributeError('PCSIM does not support populations with more than 3 dimensions')
        ##
        ##self.actual_ndim = len(dims)       
        ##while len(dims) < 3:
        ##    dims += (1,)
        ### There is a problem here, since self.dim should hold the nominal dimensions of the
        ### population, while in PCSIM the population is always really 3D, even if some of the
        ### dimensions have size 1. We should add a variable self._dims to hold the PCSIM dimensions,
        ### and make self.dims be the nominal dimensions.
        common.Population.__init__(self, dims, cellclass, cellparams, label)
                         
        ### set the steps list, used by the __getitem__() method.
        ##self.steps = [1]*self.ndim
        ##for i in range(self.ndim-1):
        ##    for j in range(i+1, self.ndim):
        ##        self.steps[i] *= self.dim[j]
        
        if isinstance(cellclass, str):
            if not cellclass in dir(pypcsim):
                raise common.InvalidModelError('Trying to create non-existent cellclass ' + cellclass )
            cellclass = getattr(pypcsim, cellclass)
            self.celltype = cellclass
        if issubclass(cellclass, common.StandardCellType):
            self.celltype = cellclass(cellparams)
            self.cellfactory = self.celltype.simObjFactory
        else:
            self.celltype = cellclass
            if issubclass(cellclass, pypcsim.SimObject):
                self.cellfactory = apply(cellclass, (), cellparams)
            else:
                raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
        
            
        # CuboidGridPopulation(SimNetwork &net, GridPoint3D origin, Volume3DSize dims, SimObjectFactory &objFactory)
        ##self.pcsim_population = pypcsim.CuboidGridObjectPopulation(
        ##                            simulator.net,
        ##                            pypcsim.GridPoint3D(0,0,0),
        ##                            pypcsim.Volume3DSize(dims[0], dims[1], dims[2]),
        ##                            self.cellfactory)
        ##self.cell = numpy.array(self.pcsim_population.idVector())
        ##self.first_id = 0
        ##self.cell -= self.cell[0]
        ##self.all_cells = self.cell
        ##self.local_cells = numpy.array(self.pcsim_population.localIndexes())
        ##
        self.all_cells, self._mask_local, self.first_id, self.last_id = simulator.create_cells(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        self.all_cells = self.all_cells.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        self.cell = self.all_cells # temporary, awaiting harmonisation
        
        self.recorders = {'spikes': simulator.Recorder('spikes', population=self),
                          'v': simulator.Recorder('v', population=self),
                          'gsyn': simulator.Recorder('gsyn', population=self)}
        
        
        if not self.label:
            self.label = 'population%d' % Population.nPop         
        Population.nPop += 1
        
        
    ##def __getitem__(self, addr):
    ##    """Return a representation of the cell with coordinates given by addr,
    ##       suitable for being passed to other methods that require a cell id.
    ##       Note that __getitem__ is called when using [] access, e.g.
    ##         p = Population(...)
    ##         p[2,3] is equivalent to p.__getitem__((2,3)).
    ##    """
    ##    if isinstance(addr, int):
    ##        addr = (addr,)
    ##    if len(addr) != self.actual_ndim:
    ##       raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.actual_ndim, str(addr))
    ##    orig_addr = addr;
    ##    while len(addr) < 3:
    ##        addr += (0,)                  
    ##    index = 0
    ##    for i, s in zip(addr, self.steps):
    ##        index += i*s 
    ##    pcsim_index = self.pcsim_population.getIndex(addr[0], addr[1], addr[2])
    ##    assert index == pcsim_index, " index = %s, pcsim_index = %s" % (index, pcsim_index)
    ##    id = ID(pcsim_index)
    ##    id.parent = self
    ##    if orig_addr != self.locate(id):
    ##        raise IndexError, 'Invalid cell address %s' % str(addr)
    ##    assert orig_addr == self.locate(id), 'index=%s addr=%s id=%s locate(id)=%s' % (index, orig_addr, id, self.locate(id))
    ##    return id
    
    ##def __iter__(self):
    ##    return self.__gid_gen()
    
    def __gid_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning gids.
        """
        ids = self.pcsim_population.idVector()
        for i in ids:
            id = ID(i-ids[0])
            id.parent = self
            yield id
        
    ##def locate(self, id):
    ##    """Given an element id in a Population, return the coordinates.
    ##           e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
    ##                     7 9
    ##    """
    ##    assert isinstance(id, ID)
    ##    if self.ndim == 3:
    ##        rows = self.dim[1]; cols = self.dim[2]
    ##        i = id/(rows*cols); remainder = id%(rows*cols)
    ##        j = remainder/cols; k = remainder%cols
    ##        coords = (i, j, k)
    ##    elif self.ndim == 2:
    ##        cols = self.dim[1]
    ##        i = id/cols; j = id%cols
    ##        coords = (i, j)
    ##    elif self.ndim == 1:
    ##        coords = (id,)
    ##    else:
    ##        raise common.InvalidDimensionsError
    ##    if self.actual_ndim == 1:
    ##        if coords[0] > self.dim[0]:
    ##            coords = None # should probably raise an Exception here rather than hope one will be raised down the line
    ##        else:
    ##            coords = (coords[0],)
    ##    elif self.actual_ndim == 2:
    ##        coords = (coords[0], coords[1],)
    ##    pcsim_coords = self.pcsim_population.getLocation(id)
    ##    pcsim_coords = (pcsim_coords.x(), pcsim_coords.y(), pcsim_coords.z())
    ##    if self.actual_ndim == 1:
    ##        pcsim_coords = (pcsim_coords[0],)
    ##    elif self.actual_ndim == 2:
    ##        pcsim_coords = (pcsim_coords[0], pcsim_coords[1],)
    ##    if coords:
    ##        assert coords == pcsim_coords, " coords = %s, pcsim_coords = %s " % (coords, pcsim_coords)
    ##    return coords
    
    ##def getObjectID(self, index):
    ##    return self.pcsim_population[index]
    
    ##def __len__(self):
    ##    """Return the total number of cells in the population."""
    ##    return self.pcsim_population.size()
        
    ##def tset(self, parametername, value_array):
    ##    """
    ##    'Topographic' set. Set the value of parametername to the values in
    ##    value_array, which must have the same dimensions as the Population.
    ##    """
    ##    """PCSIM: iteration and set """
    ##    if self.dim[0:self.actual_ndim] == value_array.shape:
    ##        values = numpy.copy(value_array) # we do not wish to change the original value_array in case it needs to be reused in user code
    ##        for cell, val in zip(self, values.flat):
    ##            cell.set_parameters(**{parametername: val})
    ##    elif len(value_array.shape) == len(self.dim[0:self.actual_ndim])+1: # the values are themselves 1D arrays
    ##        for cell,addr in zip(self.ids(), self.addresses()):
    ##            val = value_array[addr]
    ##            setattr(cell, parametername, val)
    ##    else:
    ##        raise common.InvalidDimensionsError
        
    ##def rset(self, parametername, rand_distr):
    ##    """
    ##    'Random' set. Set the value of parametername to a value taken from
    ##    rand_distr, which should be a RandomDistribution object.
    ##    """
    ##    """
    ##        Will be implemented in the future more efficiently for 
    ##        NativeRNGs.
    ##    """         
    ##    rarr = numpy.array(rand_distr.next(n=self.size))
    ##    rarr = rarr.reshape(self.dim[0:self.actual_ndim])         
    ##    self.tset(parametername, rarr)
    
    def _call(self, methodname, arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        """ This works nicely for PCSIM for simulator specific cells, 
            because cells (SimObject classes) are directly wrapped in python """
        for i in xrange(0, len(self)):
            obj = simulator.net.object(self.pcsim_population[i])
            if obj: apply( obj, methodname, (), arguments)
        
    def _tcall(self, methodname, objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init", vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i, j.
        """
        """ PCSIM: iteration at the python level and apply"""
        for i in xrange(0, len(self)):
            obj = simulator.net.object(self.pcsim_population[i])
            if obj: apply( obj, methodname, (), arguments)
        

class Projection(common.Projection, WDManager):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    nProj = 0
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 method, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
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
        
        rng - specify an RNG object to be used by the Connector..
        """
        """
           PCSIM implementation specific comments:
               - source parameter does not have any meaning in context of PyPCSIM interface. Action potential
               signals are predefined by the neuron model and each cell has only one source, 
               so there is no need to name a source since is implicitly known. 
               - rng parameter is also not currently not applicable. For connection making only internal
               random number generators can be used.
               - The semantics of the target parameter is slightly changed:
                   If it is a string then it represents a pcsim synapse class.
                   If it is an integer then it represents which target(synapse) on the postsynaptic cell
                   to connect to.
                   It can be also a pcsim SimObjectFactory object which will be used for creation 
                   of the synapse objects associated to the created connections.
                   
        """
        
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)
        if not label:
            self.label = 'projection%d' % Projection.nProj
        logging.info("--- Projection[%s].__init__() ---" %self.label)
        if not rng:
            self.rng = pyNN.random.NumpyRNG()
        self.synapse_type = target or 'excitatory'
        self.connection_manager = simulator.ConnectionManager(parent=self)
        self.connections = self.connection_manager
        method.connect(self)
        Projection.nProj += 1        
        
        self.is_conductance = "cond" in self.post.celltype.__class__.__name__ 
        
        ### Determine connection decider
        ##decider, wiring_method, weight, delay = method.connect(self)
        ##
        ##weight = self.getWeight(weight)
        ##self.is_conductance = hasattr(self.post.pcsim_population.object(0),'ErevExc')
        ##
        ##if isinstance(weight, pyNN.random.RandomDistribution) or hasattr(weight, '__len__'):
        ##    w = 1.
        ##else:
        ##    w = self.convertWeight(weight, self.is_conductance)
        ##
        ##delay  = self.getDelay(delay)
        ##if isinstance(delay, pyNN.random.RandomDistribution) or hasattr(delay, '__len__'):
        ##    d = simulator.state.min_delay/1000.
        ##else:
        ##    d = self.convertDelay(delay)
        ##
        ### handle synapse dynamics
        ##if self.synapse_dynamics:
        ##    
        ##    # choose the right model depending on whether we have conductance- or current-based synapses
        ##    if self.is_conductance:
        ##        possible_models = Set([pypcsim.DynamicStdpCondExpSynapse])
        ##    else:
        ##        possible_models = Set([pypcsim.DynamicStdpSynapse])
        ##        
        ##    plasticity_parameters = {}
        ##    
        ##    # we need to know the synaptic time constant, which is a property of the
        ##    # post-synaptic cell in PyNN. Here, we get it from the Population initial
        ##    # value, but this is a problem if tau_syn varies from cell to cell
        ##    if target in (None, 'excitatory'):
        ##        tau_syn = self.post.celltype.parameters['TauSynExc']
        ##    elif target == 'inhibitory':
        ##        tau_syn = self.post.celltype.parameters['TauSynInh']
        ##    else:
        ##        raise Exception("Currently, target must be one of 'excitatory', 'inhibitory' with dynamic synapses")
        ##
        ##    if self.synapse_dynamics.fast:
        ##        possible_models = possible_models.intersection(self.short_term_plasticity_mechanism)
        ##        plasticity_parameters.update(self._short_term_plasticity_parameters)
        ##        # perhaps need to ensure that STDP is turned off here, to be turned back on by the next block
        ##    if self.synapse_dynamics.slow:
        ##        possible_models = possible_models.intersection(self.long_term_plasticity_mechanism)
        ##        plasticity_parameters.update(self._stdp_parameters)
        ##        dendritic_delay = self.synapse_dynamics.slow.dendritic_delay_fraction * d
        ##        transmission_delay = d - dendritic_delay
        ##        plasticity_parameters.update({'back_delay': 2*dendritic_delay})
        ##        
        ##    if not self.synapse_dynamics.fast: # turn off short-term plasticity
        ##        plasticity_parameters.update({'U': 1.0, 'D': 0.0, 'F': 0.0, 'u0': 1.0, 'r0': 1.0, 'f0': -1})
        ##    assert len(possible_models) == 1, str(possible_models)
        ##    synapse_type = list(possible_models)[0]
        ##    self.syn_factory = synapse_type(Winit=w, delay=d, tau=tau_syn,
        ##                                    **plasticity_parameters)
        ##else:
        ##    if not target:
        ##        self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, w, d)
        ##    elif isinstance(target, int):
        ##        self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(target, w, d)
        ##    else:
        ##        if isinstance(target, str):
        ##            if target == 'excitatory':
        ##                self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, w, d)
        ##            elif target == 'inhibitory':
        ##                self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(2, w, d)
        ##            else:
        ##                target = eval(target)
        ##                self.syn_factory = target({})
        ##        else:
        ##            self.syn_factory = target
        ##    
        ##self.pcsim_projection = pypcsim.ConnectionsProjection(self.pre.pcsim_population, self.post.pcsim_population, 
        ##                                                      self.syn_factory, decider, wiring_method, collectIDs = True,
        ##                                                      collectPairs=True)
        ##
        ########## Should be removed and better implemented by using
        ### the fact that those random Distribution can be passed directly
        ### while the network is build, and not set after...
        ##if isinstance(weight, pyNN.random.RandomDistribution):
        ##    self.randomizeWeights(weight)
        ##elif hasattr(weight, '__len__'):
        ##    assert len(weight) == len(self), "Weight array does not have the same number of elements as the Projection %d != %d" % (len(weight),len(self))
        ##    self.setWeights(weight)
        ##
        ##if isinstance(delay, pyNN.random.RandomDistribution):
        ##    self.randomizeDelays(delay)
        ##elif hasattr(delay, '__len__'):
        ##    assert len(delay) == len(self), "Weight array does not have the same number of elements as the Projection %d != %d" % (len(weight),len(self))
        ##    self.setDelays(delay)
        ##
        ##if not self.label:
        ##    self.label = 'projection%d' % Projection.nProj
        ##if not rng:
        ##    self.rng = numpy.random.RandomState()
        ##    
        ##self.connections = simulator.ConnectionManager(parent=self)
        ##Projection.nProj += 1

    ##def __len__(self):
    ##    """Return the total number of connections."""
    ##    return self.pcsim_projection.size()
    
    #def __getitem__(self, n):
    #    return self.pcsim_projection[n]

     
    # --- Methods for setting connection parameters ----------------------------
    
    ##def setWeights(self, w):
    ##    """
    ##    w can be a single number, in which case all weights are set to this
    ##    value, or a list/1D array of length equal to the number of connections
    ##    in the population.
    ##    Weights should be in nA for current-based and µS for conductance-based
    ##    synapses.
    ##    """
    ##    w = self.convertWeight(w, self.is_conductance)
    ##    if isinstance(w, float) or isinstance(w, int):
    ##        for i in range(len(self)):
    ##            simulator.net.object(self.pcsim_projection[i]).W = w
    ##    else:
    ##        for i in range(len(self)):
    ##            simulator.net.object(self.pcsim_projection[i]).W = w[i]
    ##
    ##def randomizeWeights(self, rand_distr):
    ##    """
    ##    Set weights to random values taken from rand_distr.
    ##    """
    ##    # Arguably, we could merge this with set_weights just by detecting the
    ##    # argument type. It could make for easier-to-read simulation code to
    ##    # give it a separate name, though. Comments?
    ##    rand_distr = self.convertWeight(rand_distr, self.is_conductance)
    ##    weights = rand_distr.next(len(self))
    ##    for i in range(len(self)):
    ##        simulator.net.object(self.pcsim_projection[i]).W = weights[i]
    ## 
    ##def setDelays(self, d):
    ##    """
    ##    d can be a single number, in which case all delays are set to this
    ##    value, or a list/1D array of length equal to the number of connections
    ##    in the population.
    ##    """
    ##    # with STDP, will need updating to take account of the dendritic_delay_fraction
    ##    d = self.convertDelay(d)
    ##    if isinstance(d, float) or isinstance(d, int):
    ##        for i in range(len(self)):
    ##            simulator.net.object(self.pcsim_projection[i]).delay = d
    ##    else:
    ##        assert 1000.0*min(d) >= simulator.state.min_delay, "Smallest delay %g ms must be larger than %g ms" % (min(d), simulator.state.min_delay)
    ##        for i in range(len(self)):
    ##            simulator.net.object(self.pcsim_projection[i]).delay = d[i]
    ##
    ##def randomizeDelays(self, rand_distr):
    ##    """
    ##    Set delays to random values taken from rand_distr.
    ##    """
    ##    rand_distr = self.convertDelay(rand_distr)
    ##    delays = rand_distr.next(len(self))
    ##    for i in range(len(self)):
    ##        simulator.net.object(self.pcsim_projection[i]).delay = delays[i]
    ##
    ##def getWeights(self, format='list', gather=True):
    ##    """
    ##    Possible formats are: a list of length equal to the number of connections
    ##    in the projection, a 2D weight array (with zero or None for non-existent
    ##    connections).
    ##    """
    ##    if format == 'list':
    ##        if self.is_conductance:
    ##            A = 1e6 # S --> uS
    ##        else:
    ##            A = 1e9 # A --> nA
    ##        return [A*self.pcsim_projection.object(i).W for i in xrange(self.pcsim_projection.size())]
    ##    elif format == 'array':
    ##        raise Exception("Not yet implemented")
    ##    else:
    ##        raise Exception("Valid formats are 'list' and 'array'")
    ##
    ##def getDelays(self, format='list', gather=True):
    ##    """
    ##    Possible formats are: a list of length equal to the number of connections
    ##    in the projection, a 2D weight array (with zero or None for non-existent
    ##    connections).
    ##    """
    ##    if format == 'list':
    ##        A = 1e3 # s --> ms
    ##        return [A*self.pcsim_projection.object(i).delay for i in xrange(self.pcsim_projection.size())]
    ##    elif format == 'array':
    ##        raise Exception("Not yet implemented")
    ##    else:
    ##        raise Exception("Valid formats are 'list' and 'array'")
    ##
    ### --- Methods for writing/reading information to/from file. ----------------
    ##
    ##def saveConnections(self, filename, gather=False):
    ##    """Save connections to file in a format suitable for reading in with the
    ##    'fromFile' method."""
    ##    # Not at all sure this will work for distributed simulations
    ##    f = open(filename, 'w')
    ##    for i in range(self.pcsim_projection.size()):
    ##        pre_id, post_id = self.pcsim_projection.prePostPair(i)
    ##        pre_id = list(self.pre.pcsim_population.idVector()).index(pre_id.packed()) # surely there is an easier/faster way?
    ##        post_id = list(self.post.pcsim_population.idVector()).index(post_id.packed())
    ##        pre_addr = self.pre.locate(ID(pre_id))
    ##        post_addr = self.post.locate(ID(post_id))
    ##        w = self.reverse_convertWeight(self.pcsim_projection.object(i).W, self.is_conductance)
    ##        d = 1e3*self.pcsim_projection.object(i).delay
    ##        f.write("%s\t%s\t%g\t%g\n" % (map(int, pre_addr), map(int, post_addr), w, d))
    ##    f.close()
    


# ==============================================================================

