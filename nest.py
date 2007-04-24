# -*- coding: utf-8 -*-
"""
PyNEST implementation of the PyNN API.
$Id:nest.py 5 2007-04-16 15:01:24Z davison $
"""
__version__ = "$Revision:5 $"

import pynest
from pyNN import common
from pyNN.random import *
import numpy, types, sys, shutil, os
import logging, copy
from math import *
import tempfile


spike_files = []
v_files     = []
dt          = 0.1
tempdirs = []


# ==============================================================================
#   Utility classes
# ==============================================================================

class ID(common.ID):
    """
    This class is experimental. The idea is that instead of storing ids as
    integers, we store them as ID objects, which allows a syntax like:
      p[3,4].set('tau_m',20.0)
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def set(self,param,val=None):
        # We perform a call to the low-level function set() of the API.
        # If the cellclass is not defined in the ID object :
        if (self._cellclass == None):
            raise Exception("Unknown cellclass")
        else:
            # We use the one given by the user
            set(self,self._cellclass,param,val)

    #def get(self,param):
    #    pynest.getDict(self)[param]

# ==============================================================================
#   Standard cells
# ==============================================================================
 
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
            'v_rest'    : ('U0'    , "parameters['v_rest']"),
            'v_reset'   : ('Vreset', "parameters['v_reset']"),
            'cm'        : ('C'     , "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('Tau'   , "parameters['tau_m']"),
            'tau_refrac': ('TauR'  , "max(dt,parameters['tau_refrac'])"),
            'tau_syn'   : ('TauSyn', "parameters['tau_syn']"),
            'v_thresh'  : ('Theta' , "parameters['v_thresh']"),
            'i_offset'  : ('I0'    , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'v_init'    : ('u'     , "parameters['v_init']"),
    }
    nest_name = "iaf_neuron2"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = {
        'v_rest'    : ('U0'     , "parameters['v_rest']"),
        'v_reset'   : ('Vreset' , "parameters['v_reset']"),
        'cm'        : ('C'      , "parameters['cm']*1000.0"), # C is in pF, cm in nF
        'tau_m'     : ('Tau'    , "parameters['tau_m']"),
        'tau_refrac': ('TauR'   , "max(dt,parameters['tau_refrac'])"),
        'tau_syn_E' : ('TauSynE', "parameters['tau_syn_E']"),
        'tau_syn_I' : ('TauSynI', "parameters['tau_syn_I']"),
        'v_thresh'  : ('Theta'  , "parameters['v_thresh']"),
        'i_offset'  : ('I0'     , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
        'v_init'    : ('u'      , "parameters['v_init']"),
    }
    nest_name = 'iaf_exp_neuron2'
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)


class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'rate' : ('rate', "parameters['rate']"),
        'start' : ('start'    , "parameters['start']"),
        'duration' : ('duration' , "parameters['duration']")
    }
    nest_name = 'poisson_generator'
    
    
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['origin'] = 1.0
    
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = {
        'spike_times' : ('spike_times' , "parameters['spike_times']"),
    }
    nest_name = 'spike_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)  
    

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False):
    """Should be called at the very beginning of a script."""
    if min_delay > max_delay:
        raise Exception("min_delay has to be less than or equal to max_delay.")
    global dt
    global tempdir
    dt = timestep
    
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    
    pynest.destroy()
    pynest.setDict([0],{'resolution': dt, 'min_delay' : min_delay, 'max_delay' : max_delay})
    
    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning() 
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='nest.log',
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='nest.log',
                    filemode='w')
                       
    logging.info("Initialization of Nest")    
    return 0

def end():
    """Do any necessary cleaning up before exiting."""
    for files in spike_files:
        pynest.sr('%s close' % file)
    for couples in v_files:
        pynest.sr('%s close' % couples[1].replace('/','_'))
    for tempdir in tempdirs:
        os.rmdir(tempdir)
    
    pynest.end()

def run(simtime):
    """Run the simulation for simtime ms."""
    pynest.simulate(simtime)

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    pynest.setDict([0],{'rng_seeds': seedList})

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass,paramDict=None,n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(paramDict)
        cell_gids = pynest.create(celltype.nest_name,n)
        cell_gids = [pynest.getGID(gid) for gid in cell_gids]
        pynest.setDict(cell_gids,celltype.parameters)
    elif isinstance(cellclass, str):  # celltype is not a standard cell
        cell_gids = pynest.create(cellclass,n)
        cell_gids = [pynest.getGID(gid) for gid in cell_gids]
        if paramDict:
            pynest.setDict(cell_gids,paramDict)
    else:
        raise "Invalid cell type"
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise."""
    global dt
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = pynest.getLimits()['min_delay']
    weight = weight*1000 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                         # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            connect_id = pynest.connectWD(pynest.getAddress(source),pynest.getAddress(target),weight,delay)
        else:
            connect_id = []
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            for src in source:
                src = pynest.getAddress(src)
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = self.rng.uniform(0,1,len(target))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0,1,len(target))
                for j,tgt in enumerate(target):
                    tgt = pynest.getAddress(tgt)
                    if p >= 1 or rarr[j] < p:
                        connect_id += [pynest.connectWD(src,tgt,weight,delay)]
    except pynest.SLIError:
        raise common.ConnectionError
    return connect_id

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
        
    if val:
        param = {param:val}
    if type(cells) != types.ListType:
        cells = [cells]
    if issubclass(cellclass, common.StandardCellType):
        param = cellclass({}).translate(param)
    pynest.setDict(cells,param)

def record(source,filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    spike_detector = pynest.create('spike_detector')
    pynest.setDict(spike_detector,{'withtime':True,  # record time of spikes
                                   'withpath':True}) # record which neuron spiked
    if type(source) == types.ListType:
        source = [pynest.getAddress(src) for src in source]
    else:
        source = [pynest.getAddress(source)]
    for src in source:
        pynest.connect(src,spike_detector[0])
        pynest.sr('/%s (%s) (w) file def' % (filename,filename))
        pynest.sr('%s << /output_stream %s >> SetStatus' % (pynest.getGID(spike_detector[0]),filename))
    
    
def record_v(source,filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    if type(source) == types.ListType:
        source = [pynest.getAddress(src) for src in source]
    else:
        source = [pynest.getAddress(source)]
    for src in source:
        pynest.record_v(src,filename)
    

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
    
    def __init__(self,dims,cellclass,cellparams=None,label=None):
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
        
        common.Population.__init__(self,dims,cellclass,cellparams,label)  # move this to common.Population.__init__()
        
        # Should perhaps use "LayoutNetwork"?
        
        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cell = pynest.create(self.celltype.nest_name, self.size)
            self.cellparams = self.celltype.parameters
        elif isinstance(cellclass, str):
            self.cell = pynest.create(cellclass, self.size)
            
        self.cell = numpy.array([ ID(pynest.getGID(addr)) for addr in self.cell ], ID)
        self.id_start = self.cell.reshape(self.size,)[0]
        
        for id in self.cell:
            id.setCellClass(cellclass)
            id.setPosition(self.locate(id))
            
        if self.cellparams:
            pynest.setDict(self.cell, self.cellparams)
            
        self.cell = numpy.reshape(self.cell, self.dim)    
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        Population.nPop += 1
    
    def __getitem__(self,addr):
        """Returns a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        if isinstance(addr,int):
            addr = (addr,)
        if len(addr) == self.ndim:
            id = self.cell[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim,str(addr))
        if addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        return id
    
    def __len__(self):
        """Returns the total number of cells in the population."""
        return self.size
    
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # The top two lines (commented out) are the original implementation,
        # which does not scale well when the population size gets large.
        # The next lines are the neuron2 implementation of the same method. This
        # assumes that the id values in self.cell are consecutive. This should
        # always be the case, I think? A unit test is needed to check this.
    
        ###assert isinstance(id,int)
        ###return tuple([a.tolist()[0] for a in numpy.where(self.cell == id)])
        
        id -= self.id_start
        if self.ndim == 3:
            rows = self.dim[0]; cols = self.dim[1]
            i = id/(rows*cols); remainder = id%(rows*cols)
            j = remainder/cols; k = remainder%cols
            coords = (i,j,k)
        elif self.ndim == 2:
            cols = self.dim[1]
            i = id/cols; j = id%cols
            coords = (i,j)
        elif self.ndim == 1:
            coords = (id,)
        else:
            raise common.InvalidDimensionsError
        return coords
    
    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such.
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param,str):
            if isinstance(val,str) or isinstance(val,float) or isinstance(val,int):
                paramDict = {param:float(val)}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param,dict):
            paramDict = param
        else:
            raise common.InvalidParameterValueError
        if isinstance(self.celltype, common.StandardCellType):
            paramDict = self.celltype.translate(paramDict)
        pynest.setDict(numpy.reshape(self.cell,(self.size,)), paramDict)
        

    def tset(self,parametername,valueArray):
        """
        'Topographic' set. Sets the value of parametername to the values in
        valueArray, which must have the same dimensions as the Population.
        """
        # Convert everything to 1D arrays
        cells = numpy.reshape(self.cell,self.cell.size)
        if self.cell.shape == valueArray.shape: # the values are numbers or non-array objects
            values = numpy.reshape(valueArray,self.cell.size)
        elif len(valueArray.shape) == len(self.cell.shape)+1: # the values are themselves 1D arrays
            values = numpy.reshape(valueArray,(self.cell.size,valueArray.size/self.cell.size))
        else:
            raise common.InvalidDimensionsError, "Population: %s, valueArray: %s" % (str(cells.shape), str(valueArray.shape))
        # Translate the parameter name
        if isinstance(self.celltype, common.StandardCellType):
            parametername = self.celltype.translate({parametername: values[0]}).keys()[0]
        # Set the values for each cell
        if len(cells) == len(values):
            for cell,val in zip(cells,values):
                try:
                    if not isinstance(val,str) and hasattr(val,"__len__"):
                        val = list(val) # tuples, arrays are all converted to lists, since this is what SpikeSourceArray expects. This is not very robust though - we might want to add things that do accept arrays.
                    else:
                        pynest.setDict([cell],{parametername: val})
                except pynest.SLIError:
                    raise common.InvalidParameterValueError, "Error from SLI"
        else:
            raise common.InvalidDimensionsError
        
    
    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(self.celltype, common.StandardCellType):
            parametername = self.celltype.translate({parametername: 0.0}).keys()[0]
        if isinstance(rand_distr.rng, NativeRNG):
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            rarr = rand_distr.next(n=self.size)
            cells = numpy.reshape(self.cell,self.cell.size)
            assert len(rarr) == len(cells)
            for cell,val in zip(cells,rarr):
                try:
                    pynest.setDict([cell],{parametername: val})
                except pynest.SLIError:
                    raise common.InvalidParameterValueError
            
    def _call(self,methodname,arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise Exception("Method not yet implemented")
    
    def _tcall(self,methodname,objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init",vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        raise Exception("Method not yet implemented")

    def record(self,record_from=None,rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids (e.g., (i,j,k) tuple for a 3D population)
        of the cells to record.
        """
        global spike_files
        
        self.spike_detector = pynest.create('spike_detector')
        pynest.setDict(self.spike_detector,{'withtime':True,  # record time of spikes
                                            'withpath':True}) # record which neuron spiked
	
	fixed_list = False
	
        if record_from:
            if type(record_from) == types.ListType:
		fixed_list = True
		n_rec = len(record_from)
            elif type(record_from) == types.IntType:
                n_rec = record_from
            else:
                raise "record_from must be a list or an integer"
        else:
            n_rec = self.size
        pynest.resCons(self.spike_detector[0],n_rec)

	if (fixed_list == True):
	    for neuron in record_from:
                pynest.connect(pynest.getAddress(neuron),self.spike_detector[0])
        else:
	    for neuron in numpy.reshape(self.cell,(self.cell.size,))[0:n_rec]: # should change this to pick randomly
                pynest.connect(pynest.getAddress(neuron),self.spike_detector[0])
		
	# Open temporary output file & register file with detectors
        # This should be redone now that Eilif has implemented the pythondatum datum type
        # pynest.sr('/tmpfile_%s (tmpfile_%s) (w) file def' % (self.label,self.label)) # old
        pynest.sr('/%s.spikes (%s/%s.spikes) (w) file def' %  (self.label, tempdir, self.label))
        pynest.sr('%s << /output_stream %s.spikes >> SetStatus' % (pynest.getGID(self.spike_detector[0]),self.label))
        spike_files.append('%s.spikes' % self.label)
        
        self.n_rec = n_rec

    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        global v_files
	
	fixed_list = False
        
        if record_from:
            if type(record_from) == types.ListType:
		fixed_list = True
                n_rec = len(record_from)
            elif type(record_from) == types.IntType:
                n_rec = record_from
            else:
                raise "record_from must be a list or an integer"
        else:
            n_rec = self.size

	if (fixed_list == True):
	    for neuron in record_from:
                filename = tempdir+'/'+'%s_%d.v' % (self.label, neuron)
                v_files.append([neuron, filename])
            	pynest.record_v(pynest.getAddress(neuron),filename)
	else:
            for neuron in numpy.reshape(self.cell,(self.cell.size,))[0:n_rec]: # should change this to pick randomly
                filename = tempdir+'/'+'%s_%d.v' % (self.label, neuron)
                v_files.append([neuron, filename])
                pynest.record_v(pynest.getAddress(neuron),filename)
                
    
    
    def printSpikes(self,filename,gather=True, compatible_output=True):
        """
        Prints spike times to file in the two-column format
        "spiketime cell_id" where cell_id is the index of the cell counting
        along rows and down columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        TODO : return a numpy array?
        """
        global spike_files
        
        tempfilename = '%s.spikes' % self.label
        
        if spike_files.__contains__(tempfilename):
            pynest.sr('%s close' % tempfilename)
            spike_files.remove(tempfilename)
            
        shutil.move(tempdir+'/'+tempfilename,filename)
	if (compatible_output):
	    # Here we postprocess the file to have effectively the
            # desired format :
	    # First line: dimensions of the population
	    # Then spiketime cell_id-min(cell_id)
            f = open(filename,'r',1)
            g = open("temp",'w',1)
	    # Writing dimensions of the population:
	    g.write("# ")
	    for dimension in self.dim:
	        g.write("%d\t" %dimension)
	    g.write("\n")
	
	    # Writing spiketimes, cell_id-min(cell_id)
            padding = numpy.reshape(self.cell,self.cell.size)[0]
            lines = f.readlines()
            f.close()
            for line in lines:
                single_line = line.split("\t", 1)
                neuron = int(single_line[0][1:len(single_line[0])]) - padding
                spike_time = dt*float(single_line[1])
	        g.write("%g\t%d" %(spike_time, neuron))
	    #    for i in range(0,len(self.locate(neuron))):	
            #   	g.write("%d\t" %self.locate(neuron)[i])
	        g.write("\n")
            g.close()
            os.system("mv temp %s" %filename)
	

    def meanSpikeCount(self,gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        status = pynest.get(pynest.getGID(self.spike_detector[0]))
        n_spikes = status["events"]
        return float(n_spikes)/self.n_rec

    def randomInit(self,rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        cells = numpy.reshape(self.cell,self.cell.size)
        rvals = rand_distr.next(n=self.cell.size)
        for node, v_init in zip(cells,rvals):
            pynest.setDict([node],{'u': v_init})
    
    def print_v(self,filename,gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        """
        global v_files
        #os.system("paste %s > %s" % (" ".join(v_files), filename))
        result = open(filename,'w',1)
        dt = pynest.getNESTStatus()['resolution']
        n = int(pynest.getNESTStatus()['time']/dt)
        result.write("# dt = %f\n# n = %d\n" % (dt,n))
        if (compatible_output):
            result.write("# ")
	    for dimension in self.dim:
	       result.write("%d\t" %dimension)
	    result.write("\n")
            padding = numpy.reshape(self.cell,self.cell.size)[0]
        
                
	for couples in v_files:
            if couples[0] in self.cell:
		pynest.sr('%s close' % couples[1].replace('/','_'))
		if (compatible_output):
                    f = open(couples[1],'r',1)
                    lines = f.readlines()
                    input_tuples = []
                    for line in lines:
                        single_line = line.rstrip()
		        neuron = couples[0]
		        result.write("%s\t%d\t\n" %(single_line, neuron-padding))
                    f.close()
                    os.system("rm %s" %couples[1])
                else:
                    os.system("cat %s > %s" %(couples[1], filename))
                    os.system("rm %s" %couples[1])
        for couples in v_files:
            if couples[0] in self.cell:
                v_files.remove(couples)
        
        
    
class Projection(common.Projection):
    """
    A container for all the connections between two populations, together with
    methods to set parameters of those connections, including of plasticity
    mechanisms.
    """
    
    class ConnectionDict:
            
            def __init__(self,parent):
                self.parent = parent
    
            def __getitem__(self,id):
                """Returns a (source address,target port number) tuple."""
                assert isinstance(id, int)
                return (pynest.getAddress(self.parent._sources[id]), self.parent._targetPorts[id])
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',methodParameters=None,source=None,target=None,label=None,rng=None):
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
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,method,methodParameters,source,target,label,rng)
        
        self._targetPorts = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        connection_method = getattr(self,'_%s' % method)
        if target:
            self.nconn = connection_method(methodParameters,synapse_type=target)
        else:
            self.nconn = connection_method(methodParameters)
        assert len(self._sources) == len(self._targets) == len(self._targetPorts), "Connection error. Source and target lists are of different lengths."
        self.connection = Projection.ConnectionDict(self)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        if (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(pynest.getLimits()['min_delay'])
    
    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)
    
    def connections(self):
        """for conn in prj.connections()..."""
        for i in range(len(self)):
            yield self.connection[i]
        
    def _distance(self, presynaptic_population, postsynaptic_population, src, tgt):
        """
        Return the Euclidian distance between two cells. For the moment, we do
        a scaling between the two dimensions of the populations: the target
        population is scaled to the size of the source population."""
        dist = 0.0
        src_position = src.getPosition()
        tgt_position = tgt.getPosition()
        if (len(src_position) == len(tgt_position)):
            for i in range(0,len(src_position)):
                # We normalize the positions in each population and calculate the
                # Euclidian distance :
                #scaling = float(presynaptic_population.dim[i])/float(postsynaptic_population.dim[i])
                src_coord = float(src_position[i])
                tgt_coord = float(tgt_position[i])
            
                dist += float(src_coord-tgt_coord)*float(src_coord-tgt_coord)
        else:    
            raise Exception("Method _distance() not yet implemented for Populations with different sizes.")
        return sqrt(dist)
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None,synapse_type=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
            
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        for post in postsynaptic_neurons:
            source_list = presynaptic_neurons.tolist()
            # if self connections are not allowed, check whether pre and post are the sam
            if not allow_self_connections and post in source_list:
                target_list.remove(post)
            self._targets += [post]*len(source_list)
            self._sources += source_list
            self._targetPorts += pynest.convergentConnect(source_list,post,[1.0],[0.1])
        return len(presynaptic_neurons) * len(postsynaptic_neurons)
    
    def _oneToOne(self,synapse_type=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        if self.pre.dim == self.post.dim:
            self._sources = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
            self._targets = numpy.reshape(self.post.cell,(self.post.cell.size,))
            for pre,post in zip(self._sources,self._targets):
                pre_addr = pynest.getAddress(pre)
                post_addr = pynest.getAddress(post)
                self._targetPorts.append(pynest.connect(pre_addr,post_addr))
            return self.pre.size
        else:
            raise "Method 'oneToOne' not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes."
    
    def _fixedProbability(self,parameters,synapse_type=None):
        """
        For each pair of pre-post cells, the connection probability is constant.
        """
        allow_self_connections = True
        try:
            p_connect = float(parameters)
        except TypeError:
            p_connect = parameters['p_connect']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        npre = self.pre.size
        for post in postsynaptic_neurons:
            if self.rng:
                rarr = self.rng.uniform(0,1,(npre,))
            else:
                rarr = numpy.random.uniform(0,1,(npre,))
            source_list = numpy.compress(numpy.less(rarr,p_connect),presynaptic_neurons).tolist()
            self._targets += [post]*len(source_list)
            self._sources += source_list
            self._targetPorts += pynest.convergentConnect(source_list,post,[1.0],[0.1])
        return len(self._sources)
    
    def _distanceDependentProbability(self,parameters,synapse_type=None):
        """
        For each pair of pre-post cells, the connection probability depends on distance.
        d_expression should be the right-hand side of a valid python expression
        for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
        """
        allow_self_connections = True
        if type(parameters) == types.StringType:
            d_expression = parameters
        else:
            d_expression = parameters['d_expression']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
                   
        #raise Exception("Method not yet implemented")   
        # Here we observe the connectivity rule: if it is a probability function
        # like "exp(-d^2/2s^2)" then distance_expression should have only
        # alphanumeric characters. Otherwise, if we have characters
        # like >,<, = the connectivity rule is by itself a test.
        alphanum = True
        operators = ['<', '>', '=']
        for i in range(0,len(operators)):
            if not d_expression.find(operators[i])==-1:
                alphanum = False
                        
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        
        # We need to use the gid stored as ID, so we should modify the loop to scan the global gidlist (containing ID)
        for post in postsynaptic_neurons:
            if self.rng:
                rarr = self.rng.uniform(0,1,(self.pre.size,))
            else:
                rarr = numpy.random.uniform(0,1,(self.pre.size,))
            count = 0
            for pre in presynaptic_neurons:
                if allow_self_connections or pre != post: 
                    # calculate the distance between the two cells :
                    dist = self._distance(self.pre, self.post, pre, post)
                    distance_expression = d_expression.replace('d', '%f' %dist)
                    
                    # calculate the addresses of cells
                    pre_addr  = pynest.getAddress(pre)
                    post_addr = pynest.getAddress(post)
                    
                    if alphanum:
                        if rarr[count] < eval(distance_expression):
                            self._sources.append(pre)
                            self._targets.append(post)
                            self._targetPorts.append(pynest.connect(pre_addr,post_addr)) 
                            count = count + 1
                    elif eval(distance_expression):
                        self._sources.append(pre)
                        self._targets.append(post)
                        self._targetPorts.append(pynest.connect(pre_addr,post_addr))
    
    def _fixedNumberPre(self,parameters,synapse_type=None):
        """Each presynaptic cell makes a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
        raise Exception("Method not yet implemented")
    
    def _fixedNumberPost(self,parameters,synapse_type=None): #CHEAT CHEAT CHEAT
        """Each postsynaptic cell receives a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
        elif type(parameters) == types.DictType:
            if parameters.has_key['n']: # all cells have same number of connections
                n = parameters['n']
            elif parameters.has_key['rng']: # number of connections per cell follows a distribution
                rng = parameters['rng']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        else : # assume parameters is a rng
            rng = parameters
        
        # For now, I've just written the code for n constant        
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        for post in postsynaptic_neurons:
            post_addr = pynest.getAddress(post)
            # Reserve space for connections
            pynest.resCons(post_addr,n)
            # Make connections (I'm cheating for now, just connect the first n, rather than choosing randomly)
            for pre in presynaptic_neurons[0:n]:
                self._sources.append(pre)
                self._targetPorts.append(pynest.connect(pynest.getAddress(pre),post_addr))
    
    def _fromFile(self,parameters,synapse_type=None):
        """
        Load connections from a file.
        """    
        if type(parameters) == types.FileType:
            fileobj = parameters
            # should check here that fileobj is already open for reading
            lines = fileobj.readlines()
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
            f = open(filename,'r')
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise "Argument type not yet implemented"
        
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            single_line = single_line.split("\t", 4)
            input_tuples.append(single_line)    
        f.close()
        
        self._fromList(input_tuples, synapse_type)
        
    def _fromList(self,conn_list,synapse_type=None):
        """
        Read connections from a list of tuples,
        containing ['src[x,y]', 'tgt[x,y]', 'weight', 'delay']
        """
        # We go through those tuple and extract the fields
        for i in range(len(conn_list)):
            src    = conn_list[i][0]
            tgt    = conn_list[i][1]
            weight = eval(conn_list[i][2])
            delay  = eval(conn_list[i][3])
            src = "[%s" %src.split("[",1)[1]
            tgt = "[%s" %tgt.split("[",1)[1]
            src  = eval("self.pre%s" % src)
            tgt  = eval("self.post%s" % tgt)
                        
            pre_addr = pynest.getAddress(src)
            post_addr = pynest.getAddress(tgt)
            self._sources.append(src)
            self._targets.append(tgt)        
            self._targetPorts.append(pynest.connectWD(pre_addr,post_addr, 1000*weight, delay))

    def _2D_Gauss(self,parameters,synapse_type=None):
        """
        
        """
        def rcf_2D(parameters):
            rng = parameters['rng']
            pre_id = parameters['pre_id']
            pre_position = parameters['pre_position']
            n = parameters['n']
            sigma = parameters['sigma']
            weight = parameters['weight']
            delay = parameters['delay']
            
            phi = rng.uniform(size=n)*(2.0*pi)
            r = rng.normal(scale=sigma,size=n)
            target_position_x = numpy.floor(pre_position[1]+r*numpy.cos(phi))
            target_position_y = numpy.floor(pre_position[0]+r*numpy.sin(phi))
            target_id = []
            for syn_nr in range(len(target_position_x)):
                #print syn_nr
                try:
                    # print target_position_x[syn_nr]
                    target_id.append(self.post[(target_position_x[syn_nr],target_position_y[syn_nr])])
                    # print target_id
                except IndexError:
                    target_id.append(False)
            
            pynest.divConnect(pre_id,target_id,[weight],[delay])
        
        
        n = parameters['n']
                
        if n > 0:
            ratio_dim_pre_post = ((1.*self.pre.dim[0])/(1.*self.post.dim[0]))
            print 'ratio_dim_pre_post',ratio_dim_pre_post
            run_id = 0

            for pre in numpy.reshape(self.pre.cell,(self.pre.cell.size)):
                #print 'pre',pre
                run_id +=1
                #print 'run_id',run_id
                if numpy.mod(run_id,500) == 0:
                    print 'run_id',run_id
                
                pre_position_tmp = self.pre.locate(pre)
                parameters['pre_position'] = numpy.divide(pre_position_tmp,ratio_dim_pre_post)
                parameters['pre_id'] = pre
                #a=Projection(self.pre,self.post,'rcf_2D',parameters)
                rcf_2D(parameters)
        
    def _rcf_2D(self,parameters,synapse_type=None):
        """
        Source neuron is connected to a 2D targetd population with a spatial profile (Gauss).
        parameters should have:
        rng:
        source_position: x,y of source neuron mapped to target populatio.
        source_id: source id
        n: number of synpases
        sigma: sigma of the Gauss
        """
        rng = parameters['rng']
        pre_id = parameters['pre_id']
        pre_position = parameters['pre_position']
        n = parameters['n']
        sigma = parameters['sigma']
        weight = parameters['weight']
        delay = parameters['delay']

        phi = rng.uniform(size=n)*(2.0*pi)
        r = rng.normal(scale=sigma,size=n)
        #print 'phi',phi
        #print 'r',r
        #print 'n',n
        #print 'sigma',sigma
        
        #print 'source_id',source_id
        #print 'source_position',source_position
        #print 'pre.dim',self.pre.dim
        #print 'post.dim',self.post.dim
        
        target_position_x = numpy.floor(pre_position[1]+r*numpy.cos(phi))
        target_position_y = numpy.floor(pre_position[0]+r*numpy.sin(phi))
        #print 'target_position_x',target_position_x
        #print 'target_position_y',target_position_y
        
        target_id = []
        for syn_nr in range(len(target_position_x)):
            print syn_nr
            try:
                #print target_position_x[syn_nr]
                target_id.append(self.post[(target_position_x[syn_nr],target_position_y[syn_nr])])
                #print target_id
            except IndexError:
                target_id.append(False)
                #print syn_nr*2.
                # here the target Id's are calculated, I assume that, if the x,y does not exist in the target population and IndexError is raised
            #target_id.append(self.post.locate((target_position_x[syn_nr],target_position_y[syn_nr]))
            #except IndexError:
                # if the x,y is outside and the index error was raised, a False is put into the array/list. NEST then does not make a connection to False
             #   target_id.append(False)
        
        pynest.divConnect(pre_id,target_id,[weight],[delay])
        

    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self,w):
        """
        w can be a single number, in which case all weights are set to this
        value, or an array with the same dimensions as the Projection array.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        if type(w) == types.FloatType or type(w) == types.IntType:
            w = w*1000 # weights should be in nA or µS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                       # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
            # set all the weights from a given node at once
            for src in numpy.reshape(self.pre.cell,self.pre.cell.size):
                assert isinstance(src,int), "GIDs should be integers"
                src_addr = pynest.getAddress(src)
                n = len(pynest.getDict([src_addr])[0]['weights'])
                pynest.setDict([src_addr], {'weights' : [w]*n})
        else:
            raise Exception("Method needs changing to reflect the new API") # (w can be an array)
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        for src,port in self.connections():
            pynest.setWeight(src, port, 1000*rand_distr.next())
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or an array with the same dimensions as the Projection array.
        """
        if type(d) == types.FloatType or type(d) == types.IntType:
            # Set all the delays from a given node at once.
            for src in numpy.reshape(self.pre.cell,self.pre.cell.size):
                assert isinstance(src,int), "GIDs should be integers"
                src_addr = pynest.getAddress(src)
                n = len(pynest.getDict([src_addr])[0]['delays'])
                pynest.setDict([src_addr], {'delays' : [d]*n})
        else:
            raise Exception("Method needs changing to reflect the new API") # (d can be an array)
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        for src,port in self.connections():
            pynest.setDelay(src, port, rand_distr.next())
    
    def setThreshold(self,threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        raise Exception("Method not yet implemented")
    
    
    # --- Methods relating to synaptic plasticity ------------------------------
    
    def setupSTDP(self,stdp_model,parameterDict):
        """Set-up STDP."""
        raise Exception("Method not yet implemented")
    
    def toggleSTDP(self,onoff):
        """Turn plasticity on or off."""
        raise Exception("Method not yet implemented")
    
    def setMaxWeight(self,wmax):
        """Note that not all STDP models have maximum or minimum weights."""
        raise Exception("Method not yet implemented")
    
    def setMinWeight(self,wmin):
        """Note that not all STDP models have maximum or minimum weights."""
        raise Exception("Method not yet implemented")
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self,filename,gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        f = open(filename,'w')
        # Note unit change from pA to nA or nS to uS, depending on synapse type
        weights = [0.001*pynest.getWeight(src,port) for (src,port) in self.connections()]
        delays = [pynest.getDelay(src,port) for (src,port) in self.connections()] 
        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label,"%s",self.post.label,"%s","%f","%f")
        for i in range(len(self)):
            line = fmt  % (self.pre.locate(self._sources[i]),
                           self.post.locate(self._targets[i]),
                           weights[i],
                           delays[i])
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        file = open(filename,'w')
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,)).tolist()
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,)).tolist()
        weightArray = numpy.zeros((self.pre.size,self.post.size),dtype=float)
        for src in self._sources:
            src_addr = pynest.getAddress(src)
            pynest.sps(src_addr)
            pynest.sr('GetTargets')
            targetList = [pynest.getGID(tgt) for tgt in pynest.spp()]
            pynest.sps(src_addr)
            pynest.sr('GetWeights')
            weightList = pynest.spp()
            
            i = presynaptic_neurons.index(src)
            for tgt,w in zip(targetList,weightList):
                try:
                    j = postsynaptic_neurons.index(tgt)
                    weightArray[i][j] = w
                except ValueError: # tgt is in a different population to the current postsynaptic population
                    pass
        fmt = "%g "*len(postsynaptic_neurons) + "\n"
        for i in range(weightArray.shape[0]):
            file.write(fmt % tuple(weightArray[i]))
        file.close()
            
    
    def weightHistogram(self,min=None,max=None,nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        raise Exception("Method not yet implemented")
 
# ==============================================================================
#   Utility classes
# ==============================================================================
   
Timer = common.Timer

# ==============================================================================
