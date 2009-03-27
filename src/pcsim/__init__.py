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
import os.path
import types
import sys
import numpy
import pypcsim
from pyNN.pcsim.cells import *
from pyNN.pcsim.connectors import *
from pyNN.pcsim.synapses import *

try:
    import tables
except ImportError:
    pass
import exceptions
from datetime import datetime
import operator

from pyNN.pcsim.pcsim_globals import pcsim_globals

Set = set

def checkParams(param, val=None):
    """Check parameters are of valid types, normalise the different ways of
       specifying parameters and values by putting everything in a dict.
       Called by set() and Population.set()."""
    if isinstance(param, str):
        if isinstance(val, float) or isinstance(val, int):
            param_dict = {param:float(val)}
        elif isinstance(val, str):
            param_dict = {param:val}
        else:
            raise common.InvalidParameterValueError
    elif isinstance(param, dict):
        param_dict = param
    else:
        raise common.InvalidParameterValueError
    return param_dict

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
    
    def next(self, n=1, distribution='Uniform', parameters={'a':0,'b':1}):        
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
        
        
        
class SpikesMultiChannelRecorder(object):
    
    def __init__(self, source, filename=None, source_indices=None, gather=False, parent=None):        
        self.filename = filename
        self.gather = gather
        self.recordings = []        
        self.record(source, source_indices)
        self.parent = parent
                
    def record(self, sources, src_indices = None):
        """
            Add celllist list to the list of the cells for which spikes 
            are recorded by this spikes multi recorder
        """
        if type(sources) != types.ListType:
            sources = [sources]
        if not src_indices:
            src_indices = range(len(self.recordings), len(self.recordings) + len(sources))
        global pcsim_globals
        if type(sources) != types.ListType:
            sources = [sources]        
        for i, src in zip(src_indices, sources):
            src_id = pypcsim.SimObject.ID(src)    
            rec = pcsim_globals.net.create(pypcsim.SpikeTimeRecorder(), pypcsim.SimEngine.ID(src_id.node, src_id.eng))            
            pcsim_globals.net.connect(src, rec, pypcsim.Time.sec(0))            
            if (src_id.node == pcsim_globals.net.mpi_rank()):                
                self.recordings += [ (i, rec, src) ]
                            
    def saveSpikesH5(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + net.mpi_rank()
        try:
            h5file = tables.openFile(self.filename, mode = "w", title = "spike recordings")
        except NameError:
            raise Exception("Use of this function requires PyTables.")
        for rec_info in self.recordings:
            spikes = array([rec_ids[1]] + pcsim_globals.net.object(rec_ids[0]).getSpikeTimes())
            h5file.createArray(h5file.root, "spikes_" + str(rec_ids[1]), spikes, "")
            h5file.flush()
        h5file.close()
        
    def saveSpikesText(self, filename=None, compatible_output=True):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):    
            self.filename += ".node." + net.mpi_rank()
        f = file(self.filename, "w",10000)
        all_spikes = []
        if compatible_output:
            for i, rec, src in self.recordings:            
                spikes =  1000.0*numpy.array(pcsim_globals.net.object(rec).getSpikeTimes())
                all_spikes += zip(spikes, [ i for k in xrange(len(spikes)) ])
        else:
            for i, rec, src in self.recordings:            
                spikes =  pcsim_globals.net.object(rec).getSpikeTimes()
                all_spikes += zip( [ i for k in xrange(len(spikes)) ], spikes)
        all_spikes = sorted(all_spikes, key=operator.itemgetter(1))
        f.write("# dt = %g\n" % pcsim_globals.dt)
        if self.parent:
            f.write("# first_id = %d\n# last_id = %d\n" % (self.parent.cell[0], self.parent.cell[-1]))
        for spike in all_spikes:
            f.write("%s\t%s\n" % spike )                
        f.close()        
    
    def getSpikes(self):
        all_spikes = numpy.zeros((0,2))
        for i, rec, src in self.recordings:            
            spikes =  1000.0*numpy.array(pcsim_globals.net.object(rec).getSpikeTimes())
            spikes = spikes.reshape((len(spikes),1))
            ids = i*numpy.ones(spikes.shape)
            ids_spikes = numpy.concatenate((ids, spikes), axis=1)
            all_spikes = numpy.concatenate((all_spikes, ids_spikes), axis=0)
        return all_spikes
    
    def meanSpikeCount(self):
        count = 0
        for i, rec, src in self.recordings:
            count += pcsim_globals.net.object(rec).spikeCount()
        return count / len(self.recordings)
        
    

class FieldMultiChannelRecorder:
    
    def __init__(self, sources, filename = None, src_indices = None, gather = False, fieldname = "Vm"):        
        self.filename = filename
        self.fieldname = fieldname
        self.gather = gather
        self.recordings = []
        self.record(sources, src_indices)
                        
    def record(self, sources, src_indices = None):
        """
            Add celllist to the list of the cells for which field values
            are recorded by this field multi recorder
        """
        if type(sources) != types.ListType:
            sources = [sources]
        if not src_indices:
            src_indices = range(len(self.recordings), len(self.recordings) + len(sources))
        global pcsim_globals
        for i, src in zip(src_indices, sources):
            src_id = pypcsim.SimObject.ID(src)
            rec = pcsim_globals.net.create(pypcsim.AnalogRecorder(), pypcsim.SimEngine.ID(src_id.node, src_id.eng))
            pcsim_globals.net.connect(src, self.fieldname, rec, 0, pypcsim.Time.sec(0))
            if (src_id.node == pcsim_globals.net.mpi_rank()):
                self.recordings += [ (i, rec, src) ]
                
    def saveValuesH5(self, filename = None):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + pcsim_globals.net.mpi_rank()
        try:
            h5file = tables.openFile(filename, mode = "w", title = self.filename + " recordings")
        except NameError:
            raise Exception("Use of this function requires PyTables.")
        for i, rec, src in self.recordings:
            analog_values = array([i] + pcsim_globals.net.object(rec).getRecordedValues())
            h5file.createArray(h5file.root, self.fieldname + "_" + str(src), analog_values, "")
            h5file.flush()
        h5file.close()
        
    def saveValuesText(self, filename = None, compatible_output=True):
        if filename:
            self.filename = filename
        if (pcsim_globals.net.mpi_rank() != 0):
            self.filename += ".node." + pcsim_globals.net.net.mpi_rank()
        f = file(self.filename, "w",10000)
        all_spikes = []
        if compatible_output:
            f.write("# dt = %g\n" % pcsim_globals.dt)
            f.write("# n = %d\n" % len(pcsim_globals.net.object(self.recordings[0][1]).getRecordedValues()))
            for i, rec, src in self.recordings:
                analog_values =  pcsim_globals.net.object(rec).getRecordedValues()
                for v in analog_values:
                    f.write("%g\t%d\n" % (float(v)*1000.0, i)) # convert from mV to V
            
        else:
            for i, rec, src in self.recordings:
                analog_values =  [i] +  list(pcsim_globals.net.object(rec).getRecordedValues())
                for v in analog_values:
                    f.write("%s " % v)                
                f.write("\n")
        f.close()

    def get_v(self):
        all_vm = numpy.zeros((0,3))
        for i, rec, src in self.recordings:            
            vm = 1000.0*numpy.array(pcsim_globals.net.object(rec).getRecordedValues())
            vm = vm.reshape((len(vm),1))
            dt = get_time_step()
            t = numpy.arange(0, dt*len(vm), dt).reshape((len(vm),1))
            ids = i*numpy.ones(vm.shape)
            ids_t_vm = numpy.concatenate((ids, t, vm), axis=1)
            all_vm = numpy.concatenate((all_vm, ids_t_vm), axis=0)
        return all_vm
    

class ID(long, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object.
    
    """
    
    def __init__(self, n):
        long.__init__(n)
        common.IDMixin.__init__(self)
    
    def _pcsim_cell(self):
        if self.parent:
            pcsim_cell = self.parent.pcsim_population.object(self)
        else:
            pcsim_cell = pcsim_globals.net.object(self)
        return pcsim_cell
    
    def get_native_parameters(self):
        pcsim_cell = self._pcsim_cell()
        pcsim_parameters = {}
        if self.is_standard_cell():
            parameter_names = [D['translated_name'] for D in self.cellclass.translations.values()]
        else:
            parameter_names = [] # for native cells, is there a way to get their list of parameters?
        
        for translated_name in parameter_names:
            if hasattr(self.cellclass, 'getterMethods') and translated_name in self.cellclass.getterMethods:
                getterMethod = self.cellclass.getterMethods[translated_name]
                pcsim_parameters[translated_name] = getattr(pcsim_cell, getterMethod)()    
            else:
                try:
                    pcsim_parameters[translated_name] = getattr(pcsim_cell, translated_name)
                except AttributeError, e:
                    raise AttributeError("%s. Possible attributes are: %s" % (e, dir(pcsim_cell)))
        for k,v in pcsim_parameters.items():
            if isinstance(v, pypcsim.StdVectorDouble):
                pcsim_parameters[k] = list(v)
        return pcsim_parameters
    
    def set_native_parameters(self, parameters):
        simobj = self._pcsim_cell()
        for name, value in parameters.items():
            if hasattr(self.cellclass, 'setterMethods') and name in self.cellclass.setterMethods:
                setterMethod = self.cellclass.setterMethods[name]
                getattr(simobj, setterMethod)(value)
            else:               
                setattr(simobj, name, value)

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
            delay = pcsim_globals.minDelay
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
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    global pcsim_globals, dt
    pcsim_globals.t = 0
    pcsim_globals.dt = timestep
    pcsim_globals.minDelay = min_delay
    pcsim_globals.maxDelay = max_delay
    if pcsim_globals.constructRNGSeed is None:
        if extra_params.has_key('construct_rng_seed'):
            construct_rng_seed = extra_params['construct_rng_seed']
        else:
            construct_rng_seed = datetime.today().microsecond
        pcsim_globals.constructRNGSeed = construct_rng_seed
    if pcsim_globals.simulationRNGSeed is None:
        if extra_params.has_key('simulation_rng_seed'):
            simulation_rng_seed = extra_params['simulation_rng_seed']
        else:
            simulation_rng_seed = datetime.today().microsecond
        pcsim_globals.simulationRNGSeed = simulation_rng_seed
    if extra_params.has_key('threads'):
        pcsim_globals.net = pypcsim.DistributedMultiThreadNetwork(extra_params['threads'],
                                                                  pypcsim.SimParameter( pypcsim.Time.ms(timestep), pypcsim.Time.ms(min_delay), pypcsim.Time.ms(max_delay), pcsim_globals.constructRNGSeed, pcsim_globals.simulationRNGSeed))
    else:
        pcsim_globals.net = pypcsim.DistributedSingleThreadNetwork(pypcsim.SimParameter( pypcsim.Time.ms(timestep), pypcsim.Time.ms(min_delay), pypcsim.Time.ms(max_delay), pcsim_globals.constructRNGSeed, pcsim_globals.simulationRNGSeed))
    pcsim_globals.spikes_multi_rec = {}
    pcsim_globals.vm_multi_rec = {}
    return pcsim_globals.net.mpi_rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global pcsim_globals
    for filename, rec in pcsim_globals.vm_multi_rec.items():
        rec.saveValuesText(compatible_output=compatible_output)
    for filename, rec in pcsim_globals.spikes_multi_rec.items():
        rec.saveSpikesText(compatible_output=compatible_output)    
    pcsim_globals.vm_multi_rec = {}     
    pcsim_globals.spikes_multi_rec = {}
     

def run(simtime):
    """Run the simulation for simtime ms."""
    global pcsim_globals
    pcsim_globals.t += simtime
    pcsim_globals.net.advance(int(simtime / pcsim_globals.dt ))
    return get_current_time()

def get_current_time():
    """Return the current time in the simulation."""
    return pcsim_globals.t

def get_time_step():
    return pcsim_globals.dt
common.get_time_step = get_time_step

def get_min_delay():
    return pcsim_globals.minDelay
common.get_min_delay = get_min_delay

def num_processes():
    return pcsim_globals.net.mpi_size()

def rank():
    """Return the MPI rank."""
    return pcsim_globals.net.mpi_rank()

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, param_dict=None, n=1):
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
            cellclass = getattr(pypcsim, cellclass)
        except:
            raise AttributeError("ERROR: Trying to create non-existent cellclass " + cellclass )
    if issubclass(cellclass, common.StandardCellType):
        cellfactory = cellclass(param_dict).simObjFactory
    else:
        if issubclass(cellclass, pypcsim.SimObject):
            cellfactory = apply(cellclass, (), param_dict)
        else:
            raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
    cell_list = [ID(i) for i in pcsim_globals.net.add(cellfactory, n)]
    #cell_list = pcsim_globals.net.add(cellfactory, n)
    for id in cell_list:
        id.cellclass = cellclass
    if n == 1:
        cell_list = cell_list[0]
    return cell_list

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    global pcsim_globals
    if weight is None:  weight = 0.0
    if delay  is None:  delay = pcsim_globals.minDelay
    # Convert units
    delay = delay / 1000 # Delays in pcsim are specified in seconds
    if isinstance(target, list):
        firsttarget = target[0]
    else:
        firsttarget = target
    try:
        if hasattr(pcsim_globals.net.object(firsttarget),'ErevExc'):
            weight = 1e-6 * weight # Convert from µS to S    
        else:
            weight = 1e-9 * weight # Convert from nA to A
    except exceptions.Exception, e: # non-existent connection
        raise common.ConnectionError(e)
    # Create synapse factory
    syn_factory = 0
    if synapse_type is None:
        if weight >= 0:  # decide whether to connect to the excitatory or inhibitory response 
            syn_target_id = 1
        else:
            syn_target_id = 2
        syn_factory = pypcsim.SimpleScalingSpikingSynapse(syn_target_id, weight, delay)
    else:
        if isinstance(synapse_type, type):
            syn_factory = synapse_type
        elif isinstance(synapse_type, str):
            if synapse_type == 'excitatory':
                syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, weight, delay)
            elif synapse_type == 'inhibitory':
                syn_factory = pypcsim.SimpleScalingSpikingSynapse(2, weight, delay)
            else:
                eval('syn_factory = ' + synapse_type + '()')
            syn_factory.W = weight;
            syn_factory.delay = delay;
    # Create connections
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            connections = pcsim_globals.net.connect(source, target, syn_factory)
            return connections
        else:
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            src_popul = pypcsim.SimObjectPopulation(pcsim_globals.net, source)
            dest_popul = pypcsim.SimObjectPopulation(pcsim_globals.net, target)
            connections = pypcsim.ConnectionsProjection(src_popul, dest_popul, syn_factory, pypcsim.RandomConnections(p), pypcsim.SimpleAllToAllWiringMethod(pcsim_globals.net), True)
            return connections.idVector()
    except exceptions.TypeError, e:
        raise common.ConnectionError(e)
    except exceptions.Exception, e:
        raise common.ConnectionError(e)
    

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""   
    param_dict = checkParams(param, val)
    for cell in cells:
        cell.set_parameters(**param_dict)
    

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    global pcsim_globals
    if filename in pcsim_globals.spikes_multi_rec:
        pcsim_globals.spikes_multi_rec[filename].record(source)    
    pcsim_globals.spikes_multi_rec[filename] = SpikesMultiChannelRecorder(source, filename)
            
    

def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    global pcsim_globals
    if filename in pcsim_globals.vm_multi_rec:
        pcsim_globals.vm_multi_rec[filename].record(source)
    pcsim_globals.vm_multi_rec[filename] = FieldMultiChannelRecorder(source, filename)

            

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
        
        if isinstance(dims, int): # also allow a single integer, for a 1D population
            #print "Converting integer dims to tuple"
            dims = (dims,)
        elif len(dims) > 3:
            raise exceptions.AttributeError('PCSIM does not support populations with more than 3 dimensions')
    
        self.actual_ndim = len(dims)       
        while len(dims) < 3:
            dims += (1,)
        # There is a problem here, since self.dim should hold the nominal dimensions of the
        # population, while in PCSIM the population is always really 3D, even if some of the
        # dimensions have size 1. We should add a variable self._dims to hold the PCSIM dimensions,
        # and make self.dims be the nominal dimensions.
        common.Population.__init__(self, dims, cellclass, cellparams, label)
        
        
                 
        # set the steps list, used by the __getitem__() method.
        self.steps = [1]*self.ndim
        for i in range(self.ndim-1):
            for j in range(i+1, self.ndim):
                self.steps[i] *= self.dim[j]
        
        if isinstance(cellclass, str):
            if not cellclass in dir(pypcsim):
                raise exceptions.AttributeError('Trying to create non-existent cellclass ' + cellclass.__name__ )
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
        self.pcsim_population = pypcsim.CuboidGridObjectPopulation(pcsim_globals.net, pypcsim.GridPoint3D(0,0,0), pypcsim.Volume3DSize(dims[0], dims[1], dims[2]), self.cellfactory)
        self.cell = numpy.array(self.pcsim_population.idVector())
        self.cell -= self.cell[0]
        
        if not self.label:
            self.label = 'population%d' % Population.nPop         
        self.record_from = { 'spiketimes': [], 'vtrace': [] }        
        Population.nPop += 1
        
        
    def __getitem__(self, addr):
        """Return a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        if isinstance(addr, int):
            addr = (addr,)
        if len(addr) != self.actual_ndim:
           raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.actual_ndim, str(addr))
        orig_addr = addr;
        while len(addr) < 3:
            addr += (0,)                  
        index = 0
        for i, s in zip(addr, self.steps):
            index += i*s 
        pcsim_index = self.pcsim_population.getIndex(addr[0], addr[1], addr[2])
        assert index == pcsim_index, " index = %s, pcsim_index = %s" % (index, pcsim_index)
        id = ID(pcsim_index)
        id.parent = self
        if orig_addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        assert orig_addr == self.locate(id), 'index=%s addr=%s id=%s locate(id)=%s' % (index, orig_addr, id, self.locate(id))
        return id
    
    def __iter__(self):
        return self.__gid_gen()

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.__iter__():
            yield self.locate(i)
    
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
            
    def addresses(self):
        return self.__address_gen()
    
    def ids(self):
        return self.__iter__()
        
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        assert isinstance(id, ID)
        if self.ndim == 3:
            rows = self.dim[1]; cols = self.dim[2]
            i = id/(rows*cols); remainder = id%(rows*cols)
            j = remainder/cols; k = remainder%cols
            coords = (i, j, k)
        elif self.ndim == 2:
            cols = self.dim[1]
            i = id/cols; j = id%cols
            coords = (i, j)
        elif self.ndim == 1:
            coords = (id,)
        else:
            raise common.InvalidDimensionsError
        if self.actual_ndim == 1:
            if coords[0] > self.dim[0]:
                coords = None # should probably raise an Exception here rather than hope one will be raised down the line
            else:
                coords = (coords[0],)
        elif self.actual_ndim == 2:
            coords = (coords[0], coords[1],)
        pcsim_coords = self.pcsim_population.getLocation(id)
        pcsim_coords = (pcsim_coords.x(), pcsim_coords.y(), pcsim_coords.z())
        if self.actual_ndim == 1:
            pcsim_coords = (pcsim_coords[0],)
        elif self.actual_ndim == 2:
            pcsim_coords = (pcsim_coords[0], pcsim_coords[1],)
        if coords:
            assert coords == pcsim_coords, " coords = %s, pcsim_coords = %s " % (coords, pcsim_coords)
        return coords
    
    def getObjectID(self, index):
        return self.pcsim_population[index]
    
    def __len__(self):
        """Return the total number of cells in the population."""
        return self.pcsim_population.size()
        
    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        """PCSIM: iteration through all elements """
        param_dict = checkParams(param, val)
        for cell in self:
            cell.set_parameters(**param_dict)
        
    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        """PCSIM: iteration and set """
        if self.dim[0:self.actual_ndim] == value_array.shape:
            values = numpy.copy(value_array) # we do not wish to change the original value_array in case it needs to be reused in user code
            for cell, val in zip(self, values.flat):
                cell.set_parameters(**{parametername: val})
        else:
            raise common.InvalidDimensionsError
        
    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
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
        for i in xrange(0, len(self)):
            obj = pcsim_globals.net.object(self.pcsim_population[i])
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
            obj = pcsim_globals.net.object(self.pcsim_population[i])
            if obj: apply( obj, methodname, (), arguments)
        

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        """
          The current implementation allows only one invocation of this method per population
        """
        """ PCSIM: IMPLEMENTED by an array of recorders at python level"""
        if isinstance(record_from, int):
            if not rng:   rng = pyNN.random.RandomDistribution(rng=NativeRNG(seed = datetime.today().microsecond),
                                                               distribution='UniformInteger',
                                                               parameters=(0, len(self)-1))
            src_indices = [ int(i) for i in rng.next(record_from) ]            
        elif record_from:
            src_indices = record_from
        else:
            src_indices  = range(self.pcsim_population.size())
        sources = [ self.pcsim_population[i] for i in src_indices ]
        self.spike_rec = SpikesMultiChannelRecorder(sources, None, src_indices, parent=self)
        
    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.         
        """
        """ PCSIM: IMPLEMENTED by an array of recorders """
        if isinstance(record_from, int):             
            if not rng:   rng = pyNN.random.RandomDistribution(rng=NativeRNG(seed = datetime.today().microsecond),
                                                               distribution='UniformInteger',
                                                               parameters=(0, len(self)-1))
            src_indices = [ int(i) for i in rng.next(record_from) ]             
        elif record_from:
            src_indices = record_from
        else:
            src_indices = range(self.pcsim_population.size())
        sources = [ self.pcsim_population[i] for i in src_indices ]
        self.vm_rec = FieldMultiChannelRecorder(sources, None, src_indices)
     
    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Write spike times to file.
        
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """        
        """PCSIM: implemented by corresponding recorders at python level """
        self.spike_rec.saveSpikesText(filename, compatible_output=compatible_output)
        
        
    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        """PCSIM: will be implemented by corresponding analog recorders at python level object  """
        self.vm_rec.saveValuesText(filename, compatible_output=compatible_output)
    
    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.
        """
        return self.spike_rec.getSpikes()
    
    def get_v(self, gather=True, compatible_output=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.

        """
        return self.vm_rec.get_v()
    
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
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        """ PCSIM: can be reduced to rset() where parameterName is Vinit"""
        self.rset("v_init", rand_distr)


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
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
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
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)
        
        # Determine connection decider
        if isinstance(method, str):
            weight = None
            delay = None
            decider, wiring_method, weight, delay = method.connect(self)
        
        weight = self.getWeight(weight)
        self.is_conductance = hasattr(self.post.pcsim_population.object(0),'ErevExc')
        
        if isinstance(weight, pyNN.random.RandomDistribution) or hasattr(weight, '__len__'):
            w = 1.
        else:
            w = self.convertWeight(weight, self.is_conductance)
        
        delay  = self.getDelay(delay)
        if isinstance(delay, pyNN.random.RandomDistribution) or hasattr(delay, '__len__'):
            d = pcsim_globals.minDelay/1000.
        else:
            d = self.convertDelay(delay)

        # handle synapse dynamics
        if self.synapse_dynamics:
            
            # choose the right model depending on whether we have conductance- or current-based synapses
            if self.is_conductance:
                possible_models = Set([pypcsim.DynamicStdpCondExpSynapse])
            else:
                possible_models = Set([pypcsim.DynamicStdpSynapse])
                
            plasticity_parameters = {}
            
            # we need to know the synaptic time constant, which is a property of the
            # post-synaptic cell in PyNN. Here, we get it from the Population initial
            # value, but this is a problem if tau_syn varies from cell to cell
            if target in (None, 'excitatory'):
                tau_syn = self.post.celltype.parameters['TauSynExc']
            elif target == 'inhibitory':
                tau_syn = self.post.celltype.parameters['TauSynInh']
            else:
                raise Exception("Currently, target must be one of 'excitatory', 'inhibitory' with dynamic synapses")

            if self.synapse_dynamics.fast:
                possible_models = possible_models.intersection(self.short_term_plasticity_mechanism)
                plasticity_parameters.update(self._short_term_plasticity_parameters)
                # perhaps need to ensure that STDP is turned off here, to be turned back on by the next block
            if self.synapse_dynamics.slow:
                possible_models = possible_models.intersection(self.long_term_plasticity_mechanism)
                plasticity_parameters.update(self._stdp_parameters)
                dendritic_delay = self.synapse_dynamics.slow.dendritic_delay_fraction * d
                transmission_delay = d - dendritic_delay
                plasticity_parameters.update({'back_delay': 2*dendritic_delay})
                
            if not self.synapse_dynamics.fast: # turn off short-term plasticity
                plasticity_parameters.update({'U': 1.0, 'D': 0.0, 'F': 0.0, 'u0': 1.0, 'r0': 1.0, 'f0': -1})
            assert len(possible_models) == 1, str(possible_models)
            synapse_type = list(possible_models)[0]
            self.syn_factory = synapse_type(Winit=w, delay=d, tau=tau_syn,
                                            **plasticity_parameters)
        else:
            if not target:
                self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, w, d)
            elif isinstance(target, int):
                self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(target, w, d)
            else:
                if isinstance(target, str):
                    if target == 'excitatory':
                        self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(1, w, d)
                    elif target == 'inhibitory':
                        self.syn_factory = pypcsim.SimpleScalingSpikingSynapse(2, w, d)
                    else:
                        target = eval(target)
                        self.syn_factory = target({})
                else:
                    self.syn_factory = target
            
        self.pcsim_projection = pypcsim.ConnectionsProjection(self.pre.pcsim_population, self.post.pcsim_population, 
                                                              self.syn_factory, decider, wiring_method, collectIDs = True,
                                                              collectPairs=True)
        
        ######## Should be removed and better implemented by using
        # the fact that those random Distribution can be passed directly
        # while the network is build, and not set after...
        if isinstance(weight, pyNN.random.RandomDistribution):
            self.randomizeWeights(weight)
        elif hasattr(weight, '__len__'):
            assert len(weight) == len(self), "Weight array does not have the same number of elements as the Projection %d != %d" % (len(weight),len(self))
            self.setWeights(weight)
        
        if isinstance(delay, pyNN.random.RandomDistribution):
            self.randomizeDelays(delay)
        elif hasattr(delay, '__len__'):
            assert len(delay) == len(self), "Weight array does not have the same number of elements as the Projection %d != %d" % (len(weight),len(self))
            self.setDelays(delay)
        
        if not self.label:
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
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        w = self.convertWeight(w, self.is_conductance)
        if isinstance(w, float) or isinstance(w, int):
            for i in range(len(self)):
                pcsim_globals.net.object(self.pcsim_projection[i]).W = w
        else:
            for i in range(len(self)):
                pcsim_globals.net.object(self.pcsim_projection[i]).W = w[i]
    
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        rand_distr = self.convertWeight(rand_distr, self.is_conductance)
        weights = rand_distr.next(len(self))
        for i in range(len(self)):
            pcsim_globals.net.object(self.pcsim_projection[i]).W = weights[i]
     
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        # with STDP, will need updating to take account of the dendritic_delay_fraction
        d = self.convertDelay(d)
        if isinstance(d, float) or isinstance(d, int):
            for i in range(len(self)):
                pcsim_globals.net.object(self.pcsim_projection[i]).delay = d
        else:
            assert 1000.0*min(d) >= pcsim_globals.minDelay, "Smallest delay %g ms must be larger than %g ms" % (min(d), pcsim_globals.minDelay)
            for i in range(len(self)):
                pcsim_globals.net.object(self.pcsim_projection[i]).delay = d[i]
    
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        rand_distr = self.convertDelay(rand_distr)
        delays = rand_distr.next(len(self))
        for i in range(len(self)):
            pcsim_globals.net.object(self.pcsim_projection[i]).delay = delays[i]
    
    def getWeights(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        if format == 'list':
            if self.is_conductance:
                A = 1e6 # S --> uS
            else:
                A = 1e9 # A --> nA
            return [A*self.pcsim_projection.object(i).W for i in xrange(self.pcsim_projection.size())]
        elif format == 'array':
            raise Exception("Not yet implemented")
        else:
            raise Exception("Valid formats are 'list' and 'array'")
    
    def getDelays(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        if format == 'list':
            A = 1e3 # s --> ms
            return [A*self.pcsim_projection.object(i).delay for i in xrange(self.pcsim_projection.size())]
        elif format == 'array':
            raise Exception("Not yet implemented")
        else:
            raise Exception("Valid formats are 'list' and 'array'")
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        # Not at all sure this will work for distributed simulations
        f = open(filename, 'w')
        for i in range(self.pcsim_projection.size()):
            pre_id, post_id = self.pcsim_projection.prePostPair(i)
            pre_id = list(self.pre.pcsim_population.idVector()).index(pre_id.packed()) # surely there is an easier/faster way?
            post_id = list(self.post.pcsim_population.idVector()).index(post_id.packed())
            pre_addr = self.pre.locate(ID(pre_id))
            post_addr = self.post.locate(ID(post_id))
            w = self.reverse_convertWeight(self.pcsim_projection.object(i).W, self.is_conductance)
            d = 1e3*self.pcsim_projection.object(i).delay
            f.write("%s\t%s\t%g\t%g\n" % (map(int, pre_addr), map(int, post_addr), w, d))
        f.close()
    

# ==============================================================================
#   Utility classes
# ==============================================================================

Timer = common.Timer
     
# ==============================================================================

