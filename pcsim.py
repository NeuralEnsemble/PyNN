"""
pypcsim implementation of the PyNN API. 

    Dejan Pecevski   dejan@igi.tugraz.at
    Thomas Natschlaeger   thomas.natschlaeger@scch.at
        
    December 2006

"""
import sys

import pyNN.random
from pyNN import __path__, common
import os.path
import types
import sys
import numpy
from pypcsim import *
from tables import *
import exceptions
from datetime import datetime
import operator



# global pypcsim objects used throughout simulation
class PyPCSIM_GLOBALS:    
    net = None
    dt = None
    minDelay = None
    maxDelay = None
    constructRNGSeed = None
    simulationRNGSeed = None
    spikes_multi_rec = {}
    vm_multi_rec = {}
    pass

pcsim_globals = PyPCSIM_GLOBALS()


def checkParams(param, val=None):
    """Check parameters are of valid types, normalise the different ways of
       specifying parameters and values by putting everything in a dict.
       Called by set() and Population.set()."""
    if isinstance(param, str):
        if isinstance(val, float) or isinstance(val, int):
            paramDict = {param:float(val)}
        elif isinstance(val, str):
            paramDict = {param:val}
        else:
            raise common.InvalidParameterValueError
    elif isinstance(param, dict):
        paramDict = param
    else:
        raise common.InvalidParameterValueError
    return paramDict

# Implementation of the NativeRNG
class NativeRNG(pyNN.random.NativeRNG):
    def __init__(self,seed=None,type='MersenneTwister19937'):
        pyNN.random.AbstractRNG.__init__(self,seed)
        self.rndEngine = eval(type + '()')
        if not self.seed:
            self.seed = int(datetime.today().microsecond)
        self.rndEngine.seed(self.seed)
    
    def next(self, n=1, distribution='Uniform', parameters={'a':0,'b':1}):        
        """Return n random numbers from the distribution.
        If n is 1, return a float, if n > 1, return a numpy array,
        if n <= 0, raise an Exception."""
        distribution_type = eval(distribution + "Distribution")
        if isinstance(parameters, dict):
            dist = apply(distribution_type, (), parameters)
        else:
            dist = apply(distribution_type, tuple(parameters), {})
        values = [ dist.get(self.rndEngine) for i in xrange(n) ]
        if n == 1:
            return values[0]
        else:
            return values 
        
        
        
class SpikesMultiChannelRecorder(object):
    recordings = []  
    
    def __init__(self, source, filename = None, source_indices = None, gather = False):        
        self.filename = filename
        self.gather = gather
        self.recordings = []        
        self.record(source, source_indices)        
        
                
    def record(self, sources, src_indices = None):
        """
            Add celllist list to the list of the cells for which spikes 
            are recorded by this spikes multi recorder
        """        
        if not src_indices:
            src_indices = range(len(self.recordings), len(self.recordings) + len(sources))
        global pcsim_globals
        if type(sources) != types.ListType:
            sources = [sources]        
        for i,src in zip(src_indices, sources):
            src_id = SimObject.ID(src)    
            rec = pcsim_globals.net.create(SpikeTimeRecorder(), SimEngine.ID(src_id.node, src_id.eng))            
            pcsim_globals.net.connect(src, rec, Time.sec(0))            
            if (src_id.node == pcsim_globals.net.mpi_rank()):                
                self.recordings += [ (i, rec, src) ]
            
        
                
    def saveSpikesH5(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + net.mpi_rank()
        h5file = openFile(self.filename, mode = "w", title = "spike recordings")
        for rec_info in self.recordings:
            spikes = array([rec_ids[1]] + pcsim_globals.net.object(rec_ids[0]).getSpikeTimes())
            h5file.createArray(h5file.root, "spikes_" + str(rec_ids[1]), spikes, "")
            h5file.flush()
        h5file.close()
        
    def saveSpikesText(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):    
            self.filename += ".node." + net.mpi_rank()
        f = file(self.filename, "w")
        all_spikes = []        
        for i, rec, src in self.recordings:            
            spikes =  pcsim_globals.net.object(rec).getSpikeTimes()
            all_spikes += zip( [ i for k in xrange(len(spikes)) ], spikes)
        all_spikes = sorted(all_spikes, key=operator.itemgetter(1))
        for spike in all_spikes:
            f.write("%s %s\n" % spike )                
        f.close()        
    
    def meanSpikeCount(self):
        count = 0
        for i, rec, src in self.recordings:
            count += pcsim_globals.net.object(rec).spikeCount()
        return count / len(self.recordings)
        
    

class FieldMultiChannelRecorder:
    recordings = []  
    
    def __init__(self,sources,filename = None,src_indices = None, gather = False, fieldname = "Vm"):        
        self.filename = filename
        self.fieldname = fieldname
        self.gather = gather        
        self.record(sources, src_indices)
        
                
    def record(self, sources, src_indices = None):
        """
            Add celllist to the list of the cells for which field values
            are recorded by this field multi recorder
        """        
        if not src_indices:
            src_indices = range(len(self.recordings), len(self.recordings) + len(sources))
        global pcsim_globals
        if type(sources) != types.ListType:
            sources = [sources]        
        for i,src in zip(src_indices, sources):
            src_id = SimObject.ID(src)
            rec = pcsim_globals.net.create(AnalogRecorder(), SimEngine.ID(src_id.node, src_id.eng))
            pcsim_globals.net.connect(src, self.fieldname, rec, 0, Time.sec(0))
            if (src_id.node == pcsim_globals.net.mpi_rank()):
                self.recordings += [ (i, rec, src) ]
        
                
    def saveValuesH5(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + net.mpi_rank()
        h5file = openFile(filename, mode = "w", title = self.fielname + " recordings")
        for i, rec, src in self.recordings:
            analog_values = array([i] + pcsim_globals.net.object(rec).getRecordedValues())
            h5file.createArray(h5file.root, self.fieldname + "_" + str(src), analog_values, "")
            h5file.flush()
        h5file.close()
        
    def saveValuesText(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + net.mpi_rank()
        f = file(self.filename, "w")
        all_spikes = []
        for i, rec, src in self.recordings:
            analog_values =  [i] +  pcsim_globals.net.object(rec).getRecordedValues()            
            for v in analog_values:
                f.write("%s " % v)                
            f.write("\n")
        f.close()


# ==============================================================================
#   Standard cells   
# ==============================================================================
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {        
        'tau_m'     : ('taum'    , "parameters['tau_m']" ) ,
        'cm'        : ('Cm'      , "parameters['cm']"), 
        'v_rest'    : ('Vresting', "parameters['v_rest']"), 
        'v_thresh'  : ('Vthresh' , "parameters['v_thresh']"), 
        'v_reset'   : ('Vreset'  , "parameters['v_reset']"), 
        'tau_refrac': ('Trefract', "parameters['tau_refrac']"), 
        'i_offset'  : ('Iinject' , "parameters['i_offset']"),         
        'tau_syn'   : ('TauSyn'  , "parameters['tau_syn']"), 
        'v_init'    : ('Vinit'   , "parameters['v_init']") 
    }
    pcsim_name = "LIFCurrAlphaNeuron"    
    simObjFactory = None
    
        
    def __init__(self, parameters):
        common.IF_curr_alpha.__init__(self, parameters) # checks supplied parameters and adds default                                               # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)                
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCurrAlphaNeuron(taum     = self.parameters['taum'], 
                                                Cm       = self.parameters['Cm'], 
                                                Vresting = self.parameters['Vresting'], 
                                                Vthresh  = self.parameters['Vthresh'], 
                                                Trefract = self.parameters['Trefract'], 
                                                Iinject  = self.parameters['Iinject'], 
                                                Vinit    = self.parameters['Vinit'], 
                                                Inoise   = self.parameters['Inoise'], 
                                                TauSyn   = self.parameters['TauSyn' ])



class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
       decaying-exponential post-synaptic current. (Separate synaptic currents for
       excitatory and inhibitory synapses."""
    
    translations = {
        'tau_m'     : ('taum'   , "parameters['tau_m']"), 
        'cm'        : ('Cm'      , "parameters['cm']"),
        'v_rest'    : ('Vresting', "parameters['v_rest']"), 
        'v_thresh'  : ('Vthresh' , "parameters['v_thresh']"), 
        'v_reset'   : ('Vreset'  , "parameters['v_reset']"), 
        'tau_refrac': ('Trefract', "parameters['tau_refrac']"), 
        'i_offset'  : ('Iinject' , "parameters['i_offset']"), 
        'v_init'    : ('Vinit'   , "parameters['v_init']"), 
        'tau_syn_E' : ('TauSynExc', "parameters['tau_syn_E']"), 
        'tau_syn_I' : ('TauSynInh', "parameters['tau_syn_I']"), 
    }
    
    pcsim_name = "LIFCurrExpNeuron"    
    simObjFactory = None
    setterMethods = {}
    
    def __init__(self, parameters):
        common.IF_curr_exp.__init__(self, parameters)        
        self.parameters = self.translate(self.parameters)        
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCurrExpNeuron(taum     = self.parameters['taum'], 
                                              Cm       = self.parameters['Cm'], 
                                              Vresting = self.parameters['Vresting'], 
                                              Vthresh  = self.parameters['Vthresh'], 
                                              Trefract = self.parameters['Trefract'], 
                                              Iinject  = self.parameters['Iinject'], 
                                              Vinit    = self.parameters['Vinit'], 
                                              Inoise   = self.parameters['Inoise'], 
                                              TauSynExc = self.parameters['TauSynExc'], 
                                              TauSynInh = self.parameters['TauSynInh'])


""" Implemented not tested """
class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'start'    : ('start'  , "Time.sec(parameters['start'])"), 
        'rate'     : ('rate' , "parameters['rate']"), 
        'duration' : ('duration' , "Time.sec(parameters['duration'])")
    }
    
    pcsim_name = 'PoissonSpikeTrainGenerator'    
    simObjFactory = None
    setterMethods = {}
   
    def __init__(self, parameters):
        common.SpikeSourcePoisson.__init__(self, parameters)
        self.parameters = self.translate(self.parameters)
        self.setterMethods = {}        
        self.simObjFactory = PoissonSpikeTrainGenerator(rate = self.parameters["rate"],
                                                        start = self.parameters["start"], 
                                                        duration = self.parameters["duration"])
    
    
    
""" Implemented but not tested """
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""
    translations = {
        'spike_times' : ('spikeTimes' , "parameters['spike_times']"), 
    }
    pcsim_name = 'SpikingInputNeuron'
    simObjFactory = None
    setterMethods = {}
    
    def __init__(self, parameters):
        common.SpikeSourceArray.__init__(self, parameters)
        self.parameters = self.translate(self.parameters)
        self.setterMethods = {'spikeTimes':'setSpikeTimes' }  
        self.pcsim_object_handle = SpikingInputNeuron(spikeTimes = self.parameters['spikeTimes'])
        
                        
# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10, construct_rng_seed = None, simulation_rng_seed = None):
    """Should be called at the very beginning of a script."""
    global pcsim_globals
    pcsim_globals.dt = timestep
    pcsim_globals.minDelay = min_delay
    pcsim_globals.maxDelay = max_delay
    if pcsim_globals.constructRNGSeed is None:
        if construct_rng_seed is None:
            construct_rng_seed = datetime.today().microsecond
        pcsim_globals.constructRNGSeed = construct_rng_seed
    if pcsim_globals.simulationRNGSeed is None:
        if simulation_rng_seed is None:
            simulation_rng_seed = datetime.today().microsecond
        pcsim_globals.simulationRNGSeed = simulation_rng_seed    
    pcsim_globals.net = DistributedSingleThreadNetwork(SimParameter( Time.ms(timestep), Time.ms(min_delay), Time.ms(max_delay), 
                                        pcsim_globals.constructRNGSeed, pcsim_globals.simulationRNGSeed))
    pcsim_globals.spikes_multi_rec = {}
    pcsim_globals.vm_multi_rec = {}
    return pcsim_globals.net.mpi_rank()

def end():
     """Do any necessary cleaning up before exiting."""
     global pcsim_globals
     for filename, rec in pcsim_globals.vm_multi_rec.items():
         rec.saveValuesText()
     for filename, rec in pcsim_globals.spikes_multi_rec.items():
         rec.saveSpikesText()    
     pcsim_globals.vm_multi_rec = {}     
     pcsim_globals.spikes_multi_rec = {}
     

def run(simtime):
    """Run the simulation for simtime ms."""
    global pcsim_globals
    pcsim_globals.net.advance(int(simtime / pcsim_globals.dt ))

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    """ For pcsim this function should receive a list of two seed values, 
        one seed is used for construction of the network, and one for noise 
        generation during simulation. 
        Using same values of the construction seed ensures producing 
        the same network structure and simobjects parameters. 
        Using the same values of the simulation rng seed ensures producing 
        identical simulation results, provided that the network is the same and
        it is simulated on the same number of nodes """
    """ ISSUE: There is an issue here for PCSIM:
               the seedList should be provided before construction of the network object
               since the seeds are used in the construction for initialization.
               However the network is actually constructed in the setup method.
               So for this method to work, a constraint should be imposed 
               that this method always gets called before setup. 
               Even better solution is to put seedList as an optional argument
               in setup() instead of separate method"""               
    global pcsim_globals
            
    if len(seedList) != 2:
        raise Exception("ERROR: setRNGseeds: the seedList should be of length 2")
    if not pcsim_globals.net is None:
        raise Exception("ERROR: setRNGseeds should be called before setup() : This is a PCSIM constraint to PyNN")
    pcsim_globals.constructionRNGSeed = seedList[0]
    pcsim_globals.simulationRNGSeed = seedList[1]    
    pass

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, paramDict=None, n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    global pcsim_globals
    if pcsim_globals.net is None:
        setup()
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, str):
        try:
            cellclass = eval(cellclass)
        except:
            raise AttributeError("ERROR: Trying to create non-existent cellclass " + cellclass.__name__ )
    if issubclass(cellclass, common.StandardCellType):
        cellfactory = cellclass(paramDict).simObjFactory
    else:
        if issubclass(cellclass, SimObject):
            cellfactory = apply(cellclass, (), paramDict)
        else:
            raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
    cell_list = pcsim_globals.net.add(cellfactory, n)
    if n == 1:
        cell_list = cell_list[0]
    return cell_list

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise."""
    global pcsim_globals
    if weight is None:  weight = 0.0
    if delay  is None:  delay = pcsim_globals.minDelay
    delay = delay / 1000 # Delays in pcsim are specified in seconds
    syn_factory = 0
    if synapse_type is None:
        if weight >= 0:  # decide whether to connect to the excitatory or inhibitory response 
            syn_target_id = 1
        else:
            syn_target_id = 2
        syn_factory = SimpleScalingSpikingSynapse(syn_target_id, weight, delay)
    else:
        if isinstance(synapse_type, type):
            syn_factory = synapse_type
        elif isinstance(synapse_type, str):
            eval('syn_factory = ' + synapse_type + '()')
            syn_factory.W = weight;
            syn_factory.delay = delay;
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            connections = pcsim_globals.net.connect(source, target, syn_factory)
            return connections
        else:
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            src_popul = SimObjectPopulation(pcsim_globals.net, source)
            dest_popul = SimObjectPopulation(pcsim_globals.net, target)
            connections = ConnectionsProjection(src_popul, dest_popul, syn_factory, RandomConnections(p), SimpleAllToAllWiringMethod(pcsim_globals.net), True)
            return connections.idVector()
    except exceptions.TypeError, e:
        raise common.ConnectionError(e)
    except exceptions.Exception, e:
        raise common.ConnectionError(e)
    

def set(cells, cellclass, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    global pcsim_globals    
    paramDict = checkParams(param, val)
    if issubclass(cellclass, common.StandardCellType):        
        paramDict = cellclass({}).translate(paramDict)
    for param, value in paramDict.items():
        if param in cellclass.setterMethods:
           setterMethod = cellclass.setterMethods[param]
           for id in cells:
               simobj = pcsim_globals.net.object(id)
               getattr(simobj, setterMethod)( value )
        else:            
            for id in cells:
                simobj = pcsim_globals.net.object(id)                
                setattr( simobj, param, value )
    

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    global pcsim_globals
    if filename in pcsim_globals.spikes_multi_rec:
        pcsim_globals.spikes_multi_rec[filename].record(source)    
    pcsim_globals.spikes_multi_rec[filename] = SpikesMultiChannelRecorder(source,filename)
            
    

def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    global pcsim_globals
    if filename in pcsim_globals.vm_multi_rec:
        pcsim_globals.vm_multi_rec[filename].record(source)
    pcsim_globals.vm_multi_rec[filename] = FieldMultiChannelRecorder(source,filename)

            

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
         global gid, myid, nhost
         
         if len(dims) > 3:
             raise exceptions.AttributeError('PCSIM does not support populations with more than 3 dimensions')
         
         self.actual_ndim = len(dims)
         
         while len(dims) < 3:
             dims += (1,)
         
         common.Population.__init__(self, dims, cellclass, cellparams, label)
         
         
                  
         # set the steps list, used by the __getitem__() method.
         self.steps = [1]*self.ndim
         for i in range(self.ndim-1):
             for j in range(i+1, self.ndim):
                 self.steps[i] *= self.dim[j]
         
         if isinstance(cellclass, str):
             if not cellclass in globals():
                 raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
             cellclass = eval(cellclass)
         self.celltype = cellclass
         if issubclass(cellclass, common.StandardCellType):
             self.cellfactory = cellclass(cellparams).simObjFactory
         else:
             if issubclass(cellclass, SimObject):
                 self.cellfactory = apply(cellclass, (), cellparams)
             else:
                 raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
         
             
         # CuboidGridPopulation(SimNetwork &net, GridPoint3D origin, Volume3DSize dims, SimObjectFactory &objFactory)
         self.pcsim_population = CuboidGridObjectPopulation(pcsim_globals.net, GridPoint3D(0,0,0), Volume3DSize(dims[0], dims[1], dims[2]), self.cellfactory)
         
         if not self.label:
             self.label = 'population%d' % Population.nPop         
         self.record_from = { 'spiketimes': [], 'vtrace': [] }        
         Population.nPop += 1
         
         
     def __getitem__(self, addr):
         """Returns a representation of the cell with coordinates given by addr,
            suitable for being passed to other methods that require a cell id.
            Note that __getitem__ is called when using [] access, e.g.
              p = Population(...)
              p[2,3] is equivalent to p.__getitem__((2,3)).
         """
         # What we actually pass around are gids.
         orig_addr = addr;
         if isinstance(addr, int):
             addr = (addr,)
         while len(addr) < 3:
             addr += (0,)                  
         assert len(addr) == len(self.dim)
         
         index = 0
         for i, s in zip(addr, self.steps):
             index += i*s
         id = index 
         pcsim_index = self.pcsim_population.getIndex(addr[0],addr[1],addr[2])
         assert id == pcsim_index, " id = %s, pcsim_index = %s" % (id, pcsim_index)
         assert orig_addr == self.locate(id), 'index=%s addr=%s id=%s locate(id)=%s' % (index, orig_addr, id, self.locate(id))
         return id
         
         
     def locate(self, id):
         """Given an element id in a Population, return the coordinates.
                e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                          7 9
         """
         # id should be a gid
         assert isinstance(id, int)         
         if self.ndim == 3:
             rows = self.dim[0]; cols = self.dim[1]
             i = id/(rows*cols); remainder = id%(rows*cols)
             j = remainder/cols; k = remainder%cols
             coords = (k, j, i)
         elif self.ndim == 2:
             cols = self.dim[1]
             i = id/cols; j = id%cols
             coords = (i, j)
         elif self.ndim == 1:
             coords = (id,)
         else:
             raise common.InvalidDimensionsError
         if self.actual_ndim == 1:
             coords = (coords[0],)
         elif self.actual_ndim == 2:
             coords = (coords[0],coords[1],)
         pcsim_coords = self.pcsim_population.getLocation(id)
         pcsim_coords = (pcsim_coords.x(), pcsim_coords.y(), pcsim_coords.z())
         if self.actual_ndim == 1:
             pcsim_coords = (pcsim_coords[0],)
         elif self.actual_ndim == 2:
             pcsim_coords = (pcsim_coords[0],pcsim_coords[1],)    
         # assert coords == pcsim_coords, " coords = %s, pcsim_coords = %s " % (coords, pcsim_coords)
         return pcsim_coords
     
     def getObjectID(self, index):
         return self.pcsim_population[index]
     
     def __len__(self):
         """Returns the total number of cells in the population."""
         return self.pcsim_population.size()
         
     def set(self, param, val=None):
         """PCSIM: iteration through all elements """
         """
         Set one or more parameters for every cell in the population. param
         can be a dict, in which case val should not be supplied, or a string
         giving the parameter name, in which case val is the parameter value.
         e.g. p.set("tau_m",20.0).
              p.set({'tau_m':20,'v_rest':-65})
         """
         paramDict = checkParams(param, val)
         if issubclass(self.celltype, common.StandardCellType):
             paramDict = self.celltype({}).translate(paramDict)
                  
         for index in range(0,len(self)):
             obj = pcsim_globals.net.object(self.pcsim_population[index])
             if obj:
                 for param,value in paramDict.items():
                     setattr( obj, param, value )
         
         
     def tset(self, parametername, valueArray):
         """PCSIM: iteration and set """
         """
         'Topographic' set. Sets the value of parametername to the values in
         valueArray, which must have the same dimensions as the Population.
         """
         if self.dim[0:self.actual_ndim] == valueArray.shape:
             values = numpy.reshape(valueArray, valueArray.size)                          
             if issubclass(self.celltype, common.StandardCellType):
                 parametername = self.celltype({}).translate({parametername: values[0]}).keys()[0]             
             for i, val in enumerate(values):
                 try:
                     obj = pcsim_globals.net.object(self.pcsim_population[i])                 
                     if obj: setattr(obj, parametername, val)
                 except TypeError:
                     raise common.InvalidParameterValueError, "%s is not a numeric value" % str(val)             
         else:
             raise common.InvalidDimensionsError
         
     def rset(self, parametername, rand_distr):
         """
         'Random' set. Sets the value of parametername to a value taken from
         rand_distr, which should be a RandomDistribution object.
         """
         """
             Will be implemented in the future more efficiently for 
             NativeRNGs.
         """         
         rarr = numpy.array(rand_distr.next(n=self.size))
         rarr = rarr.reshape(self.dim[0:self.actual_ndim])         
         self.tset(parametername, rarr)
     
     def _call(self, methodname, arguments):
         """
         Calls the method methodname(arguments) for every cell in the population.
         e.g. p.call("set_background","0.1") if the cell class has a method
         set_background().
         """
         """ This works nicely for PCSIM for simulator specific cells, 
             because cells (SimObject classes) are directly wrapped in python """
         for i in xrange(0,len(self)):
             obj = pcsim_globals.net.object(self.pcsim_population[i])
             if obj: apply( obj, methodname, (), arguments)
         
         
     
     def _tcall(self, methodname, objarr):
         """ PCSIM: iteration at the python level and apply"""
         """
         `Topographic' call. Calls the method methodname() for every cell in the 
         population. The argument to the method depends on the coordinates of the
         cell. objarr is an array with the same dimensions as the Population.
         e.g. p.tcall("memb_init",vinitArray) calls
         p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
         """
         for i in xrange(0,len(self)):
             obj = pcsim_globals.net.object(self.pcsim_population[i])
             if obj: apply( obj, methodname, (), arguments)
         

     def record(self, record_from=None, rng=None):
         """ PCSIM: IMPLEMENTED by an array of recorders at python level"""
         """
         If record_from is not given, record spikes from all cells in the Population.
         record_from can be an integer - the number of cells to record from, chosen
         at random (in this case a random number generator can also be supplied)
         - or a list containing the ids (e.g., (i,j,k) tuple for a 3D population)
         of the cells to record.
         """         
         """
           The current implementation allows only one invocation of this method per population
         """
         if isinstance(record_from, int):
             if not rng:   rng = pyNN.random.RandomDistribution(NativeRNG(seed = datetime.today().microsecond), 'UniformInteger', (0,len(self)-1))             
             src_indices = [ int(i) for i in rng.next(record_from) ]            
         elif record_from:
             src_indices = record_from
         else:
             src_indices  = range(self.pcsim_population.size())
         sources = [ self.pcsim_population[i] for i in src_indices ]
         self.spike_rec = SpikesMultiChannelRecorder(sources, None, src_indices)
         
     def record_v(self, record_from=None, rng=None):
         """ PCSIM: IMPLEMENTED by an array of recorders """
         """
         If record_from is not given, record the membrane potential for all cells in
         the Population.
         record_from can be an integer - the number of cells to record from, chosen
         at random (in this case a random number generator can also be supplied)
         - or a list containing the ids of the cells to record.         
         """
         if isinstance(record_from, int):             
             if not rng:   rng = pyNN.random.RandomDistribution(NativeRNG(seed = datetime.today().microsecond), 'UniformInteger', (0,len(self)-1))            
             src_indices = [ int(i) for i in rng.next(record_from) ]             
         elif record_from:
             src_indices = record_from
         else:
             src_indices = range(self.pcsim_population.size())
         sources = [ self.pcsim_population[i] for i in src_indices ]
         self.vm_rec = FieldMultiChannelRecorder(sources, None, src_indices)
     
     def printSpikes(self, filename, gather=True):
         """PCSIM: implemented by corresponding recorders at python level """
         """
         Prints spike times to file in the two-column format
         "spiketime cell_id" where cell_id is the index of the cell counting
         along rows and down columns (and the extension of that for 3-D).
         This allows easy plotting of a `raster' plot of spiketimes, with one
         line for each cell. This method requires that the cell class records
         spikes in a vector spiketimes.
         If gather is True, the file will only be created on the master node,
         otherwise, a file will be written on each node.
         """
         self.spike_rec.saveSpikesText(filename)
         
         
     def print_v(self, filename, gather=True):
         """PCSIM: will be implemented by corresponding analog recorders at python level object  """
         """
         Write membrane potential traces to file.
         """
         self.spike_rec.saveValuesText(filename)
         
     
     def meanSpikeCount(self, gather=True):         
         """
             Returns the mean number of spikes per neuron.
             NOTE: This method works in PCSIM only if you invoke the record
                   during setup of the population. And the mean spike count
                   takes into account only the neurons that are recorded, not all neurons.
                   Implemented in this way because cells in PCSIM don't have
                   actual internal recording mechanisms. All recordings are done with 
                   SpikeTimeRecorder SimObjects and spike messages between cells and 
                   recorders. 
         """
         if self.spike_rec:
             return self.spike_rec.meanSpikeCount()
         return 0;

     def randomInit(self, rand_distr):
         """ PCSIM: can be reduced to rset() where parameterName is Vinit"""
         """
         Sets initial membrane potentials for all the cells in the population to
         random values.
         """         
         self.rset("v_init", rand_distr)
     


class Projection(common.Projection):
     """
     A container for all the connections between two populations, together with
     methods to set parameters of those connections, including of plasticity
     mechanisms.
     """
     
     nProj = 0
     
     #class ConnectionDict:
     #        
     #    def __init__(self,parent):
     #        self.parent = parent
     #
     #    def __getitem__(self,id):
     #        """Returns a connection id.
     #        Suppose we have a 2D Population (5x3) projecting to a 3D Population (4x5x7).
     #        Total number of possible connections is 5x3x4x5x7 = 2100.
     #        Therefore valid calls are:
     #        connection[2099] - 2099th possible connection (may not exist)
     #        connection[14,139] - connection between 14th pre- and 139th postsynaptic neuron (may not exist)
     #        connection[(4,2),(3,4,6)] - connection between presynaptic neuron with address (4,2)
     #        and post-synaptic neuron with address (3,4,6) (may not exist).
     #        """
     #        if isinstance(id, int): # linear mapping
     #            preID = id/self.parent.post.size; postID = id%self.parent.post.size
     #            return self.__getitem__((preID,postID))
     #        elif isinstance(id, tuple): # (pre,post)
     #            if len(id) == 2:
     #                pre = id[0]
     #                post = id[1]
     #                if isinstance(pre,int) and isinstance(post,int):
     #                    pre_coords = self.parent.pre.locate(pre)
     #                    post_coords = self.parent.post.locate(post)
     #                    return self.__getitem__((pre_coords,post_coords))
     #                elif isinstance(pre,tuple) and isinstance(post,tuple): # should also allow lists
     #                    if len(pre) == self.parent.pre.ndim and len(post) == self.parent.post.ndim:
     #                        fmt = "[%d]"*(len(pre)+len(post))
     #                        address = fmt % (pre+post)
     #                    else:
     #                        raise common.InvalidDimensionsError
     #                else:
     #                    raise KeyError
     #            else:
     #                raise common.InvalidDimensionsError
     #        else:
     #            raise KeyError #most appropriate?
     #        
     #        return address
     #
     
     def __init__(self, presynaptic_population, postsynaptic_population, method='allToAll', methodParameters=None, source=None, target=None, label=None, rng=None):
         """
         presynaptic_population and postsynaptic_population - Population objects.
         
         source - string specifying which attribute of the presynaptic cell signals action potentials
         
         target - string specifying which synapse on the postsynaptic cell to connect to
         If source and/or target are not given, default values are used.
         
         method - string indicating which algorithm to use in determining connections.
         Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
         'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
         'fromFile', 'fromList'
         
         methodParameters - dict containing parameters needed by the connection method,
         although we should allow this to be a number or string if there is only
         one parameter.
         
         rng - since most of the connection methods need uniform random numbers,
         it is probably more convenient to specify a RNG object here rather
         than within methodParameters, particularly since some methods also use
         random numbers to give variability in the number of connections per cell.
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
         global pcsim_globals
         common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method, methodParameters, source, target, label, rng)
         
         # Determine connection decider
         if method == 'allToAll':
             decider = RandomConnections(1)
             wiring_method = DistributedSyncWiringMethod(pcsim_globals.net)
         elif method == 'fixedProbability':
             decider = RandomConnections(float(methodParameters))
             wiring_method = DistributedSyncWiringMethod(pcsim_globals.net)
         elif method == 'distanceDependentProbability':
             decider = EuclideanDistanceRandomConnections(methodParameters[0], methodParameters[1])
             wiring_method = DistributedSyncWiringMethod(pcsim_globals.net)
         elif method == 'fixedNumberPre':
             decider = DegreeDistributionConnections(ConstantNumber(parameters), DegreeDistributionConnections.incoming)
             wiring_method = SimpleAllToAllWiringMethod(pcsim_globals.net)
         elif method == 'fixedNumberPost':
             decider = DegreeDistributionConnections(ConstantNumber(parameters), DegreeDistributionConnections.outgoing)
             wiring_method = SimpleAllToAllWiringMethod(pcsim_globals.net)
         elif method == 'oneToOne':
             decider = RandomConnections(1)
             wiring_method = OneToOneWiringMethod(pcsim_globals.net) 
         else:
             raise Exception("METHOD NOT YET IMPLEMENTED")
             
         if not target:
             self.syn_factory = SimpleScalingSpikingSynapse(1, 1, pcsim_globals.minDelay/1000)
         elif isinstance(target, int):
             self.syn_factory = SimpleScalingSpikingSynapse(target, 1, pcsim_globals.minDelay/1000)
         else:
             if isinstance(target, str):
                 target = eval(target)
                 self.syn_factory = target({})
             else:
                 self.syn_factory = target
             
         self.pcsim_projection = ConnectionsProjection(self.pre.pcsim_population, self.post.pcsim_population, 
                                                       self.syn_factory, decider, wiring_method, collectIDs = True)

         if not label:
             self.label = 'projection%d' % Projection.nProj
         if not rng:
             self.rng = numpy.random.RandomState()
         Projection.nProj += 1

     def __len__(self):
         """Return the total number of connections."""
         return self.pcsim_projection.size()
     
     def __getitem__(self, n):
         return self.pcsim_projection[n]

     
     # --- Methods for setting connection parameters ----------------------------
     
     def setWeights(self, w):
         """
         w can be a single number, in which case all weights are set to this
         value, or an array with the same dimensions as the Projection array.
         """
         if isinstance(w, float) or isinstance(w, int):
             for i in range(len(self)):
                 pcsim_globals.net.object(self.pcsim_projection[i]).W = w
         else:
             for i in range(len(self)):
                 pcsim_globals.net.object(self.pcsim_projection[i]).W = w[i]
     
     def randomizeWeights(self, rng):
         """
         Set weights to random values taken from rng.
         """
         # Arguably, we could merge this with set_weights just by detecting the
         # argument type. It could make for easier-to-read simulation code to
         # give it a separate name, though. Comments?
         weights = rng.next(len(self))
         self.setWeights(weights)
     
     def setDelays(self, d):
         """
         d can be a single number, in which case all delays are set to this
         value, or an array with the same dimensions as the Projection array.
         """
         raise Exception("METHOD NOT YET IMPLEMENTED!")
     
     def randomizeDelays(self, rng):
         """
         Set delays to random values taken from rng.
         """
         raise Exception("Method not yet implemented!")
     
     def setThreshold(self, threshold):
         """
         Where the emission of a spike is determined by watching for a
         threshold crossing, set the value of this threshold.
         """
         # This is a bit tricky, because in NEST and PCSIM the spike threshold is a
         # property of the cell model, whereas in NEURON it is a property of the
         # connection (NetCon).
         raise Exception("Method  not applicable to PCSIM")
     
     
     # --- Methods relating to synaptic plasticity ------------------------------
     
     def setupSTDP(self, stdp_model, parameterDict):
         """Set-up STDP."""
         raise Exception("Method not yet implemented")
     
     def toggleSTDP(self, onoff):
         """Turn plasticity on or off."""
         raise Exception("Method not yet implemented")
     
     def setMaxWeight(self, wmax):
         """Note that not all STDP models have maximum or minimum weights."""
         raise Exception("Method not yet implemented")
     
     def setMinWeight(self, wmin):
         """Note that not all STDP models have maximum or minimum weights."""
         raise Exception("Method not yet implemented")
     
     # --- Methods for writing/reading information to/from file. ----------------
     
     def saveConnections(self, filename):
         """Save connections to file in a format suitable for reading in with the
         'fromFile' method."""
         # should think about adding a 'gather' option.
         raise Exception("Method not yet implemented")
         
     
     def printWeights(self, filename, format=None):
         """Print synaptic weights to file."""
         raise Exception("Method not yet implemented")
     
     def weightHistogram(self, min=None, max=None, nbins=10):
         """
         Return a histogram of synaptic weights.
         If min and max are not given, the minimum and maximum weights are
         calculated automatically.
         """
         # it is arguable whether functions operating on the set of weights
         # should be put here or in an external module.
         raise Exception("Method not yet implemented")
     
# END

# ==============================================================================
#   Utility classes
# ==============================================================================

Timer = common.Timer
     
# ==============================================================================
     
