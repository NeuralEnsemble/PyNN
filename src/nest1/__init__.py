# -*- coding: utf-8 -*-
"""
PyNEST implementation of the PyNN API.
$Id:nest1.py 143 2007-10-05 14:20:16Z apdavison $
"""
__version__ = "$Rev$"

import pynest
from pyNN import common, recording
from pyNN.random import *
import numpy, types, sys, shutil, os, logging, copy, tempfile
from math import *
from pyNN.nest1.cells import *
from pyNN.nest1.connectors import *
from pyNN.nest1.synapses import *

recorders  = {}
tempdirs   = []

DEFAULT_BUFFER_SIZE = 10000

# ==============================================================================
#   Utility classes and functions
# ==============================================================================

class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)

    def get_native_parameters(self):
        return pynest.getDict([int(self)])[0]

    def set_native_parameters(self, parameters):
        pynest.setDict([self], parameters)
       
        
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

def is_number(n):
    return type(n) == types.FloatType or type(n) == types.IntType or type(n) == numpy.float64

def _convertWeight(w, synapse_type):
    weight = w*1000.0
    if isinstance(w, numpy.ndarray):
        all_negative = (weight<=0).all()
        all_positive = (weight>=0).all()
        assert all_negative or all_positive, "Weights must be either all positive or all negative"
        if synapse_type == 'inhibitory':
            if all_positive:
                weights *= -1
    elif is_number(weight):
        if synapse_type == 'inhibitory' and weight > 0:
            weight *= -1
    else:
        raise TypeError("we must be either a number or a numpy array")
    return weight
    

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
    global tempdir
    
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    
    pynest.destroy()
    pynest.setDict([0],{'resolution': timestep, 'min_delay' : min_delay, 'max_delay' : max_delay})
    if extra_params.has_key('threads'):
        if extra_params.has_key('kernelseeds'):
            print 'params has kernelseeds ', extra_params['kernelseeds']
            kernelseeds = extra_params['kernelseeds']
        else:
            rng = NumpyRNG(42)
            kernelseeds = (rng.rng.uniform(size=extra_params['threads'])*100).astype('int').tolist()
            print 'params has no kernelseeds, we use ', kernelseeds
        update_modes = {'fixed':1, 'serial':3, 'dynamic':0}
        # number of nodes to give to each thread at a time
        # some small fraction of your total nodes to be simulated
        batchsize   = 10
        pynest.setDict([0],{'threads'     : extra_params['threads'],
                            'update_mode' : update_modes['fixed'],
                            'rng_seeds'   : kernelseeds[0:extra_params['threads']],
                            'buffsize'    : batchsize})
    
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

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    global tempdir
    # We close the high level files opened by populations objects
    # that may have not been written.
    # And we postprocess the low level files opened by record()
    # and record_v() method
    for key, value in zip(recorders.keys(), recorders.values()):
        if value[0] == "spikes":
            _printSpikes(value[1], key, compatible_output)
        if value[0] == "v":
            _print_v(value[1], key, compatible_output)
    for tempdir in tempdirs:
        os.system("rm -rf %s" %tempdir)
    pynest.end()

def run(simtime):
    """Run the simulation for simtime ms."""
    pynest.simulate(simtime)

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    pynest.setDict([0],{'rng_seeds': seedList})

def get_min_delay():
    return pynest.getLimits()['min_delay']
common.get_min_delay = get_min_delay

def get_time_step():
    return pynest.getNESTStatus()['resolution']
common.get_time_step = get_time_step

def get_current_time():
    return pynest.getNESTStatus()['time']

def num_processes():
    return 1

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, param_dict=None, n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(param_dict)
        cell_gids = pynest.create(celltype.nest_name, n)
        cell_gids = [ID(pynest.getGID(gid)) for gid in cell_gids]
        pynest.setDict(cell_gids, celltype.parameters)
    elif isinstance(cellclass, str):  # celltype is not a standard cell
        cell_gids = pynest.create(cellclass, n)
        cell_gids = [ID(pynest.getGID(gid)) for gid in cell_gids]
        if param_dict:
            pynest.setDict(cell_gids, param_dict)
    else:
        raise "Invalid cell type"
    for id in cell_gids:
    #    #id.setCellClass(cellclass)
        id.cellclass = cellclass
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = get_min_delay()
    weight = weight*1000 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                         # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
    if synapse_type == 'inhibitory' and weight > 0:
        weight *= -1
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            connect_id = pynest.connectWD(pynest.getAddress(source), pynest.getAddress(target), weight, delay)
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
                        rarr = rng.rng.uniform(0, 1, len(target))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0, 1, len(target))
                for j,tgt in enumerate(target):
                    tgt = pynest.getAddress(tgt)
                    if p >= 1 or rarr[j] < p:
                        connect_id += [pynest.connectWD(src, tgt, weight, delay)]
    except pynest.SLIError:
        raise common.ConnectionError
    return connect_id

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value."""
    # we should just assume that cellclass has been defined and raise an Exception if it has not
    if val:
        param = {param:val}
    try:
        i = cells[0]
    except TypeError:
        cells = [cells]
    if not isinstance(cellclass, str):
        if issubclass(cellclass, common.StandardCellType):
            param = cellclass({}).translate(param)
        else:
            raise TypeError, "cellclass must be a string or derived from commonStandardCellType"
    pynest.setDict(cells, param)

def record(source, filename):
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
        pynest.connect(src, spike_detector[0])
        tmpfile = "%s/%s" %(tempdir, filename)
        pynest.sr('/%s (%s) (w) file def' % (filename, tmpfile))
        pynest.sr('%s << /output_stream %s >> SetStatus' % (pynest.getGID(spike_detector[0]), filename))
    recorders[filename] = ("spikes", tmpfile)


def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    if type(source) == types.ListType:
        source = [pynest.getAddress(src) for src in source]
    else:
        source = [pynest.getAddress(source)]
    record_file = tempdir+'/'+filename
    tmpfile = record_file.replace('/','_')
    recorders[filename] = ("v", tmpfile)
    pynest.record_v(source, tmpfile)


def _printSpikes(tmpfile, filename, compatible_output=True):
    """ Print spikes into a file, and postprocessed them if
    needed and asked to produce a compatible output for all the simulator
    Should actually work with record() and allow to dissociate the recording of the
    writing process, which is not the case for the moment"""
    pynest.sr('%s close' %tmpfile[15:1000]) 
    if (compatible_output):
        # Here we postprocess the file to have effectively the
        # desired format :
        # First line: # dimensions of the population
        # Then spiketime (in ms) cell_id-min(cell_id)
        result = open(filename,'w',DEFAULT_BUFFER_SIZE)
        # Writing # such that Population.printSpikes and this have same output format
        result.write("# "+"\n")
        # Writing spiketimes, cell_id-min(cell_id)
        # Pylab has a great load() function, but it is not necessary to import
        # it into pyNN. The fromfile() function of numpy has trouble on several
        # machine with Python 2.5, so that's why a dedicated _readArray function
        # has been created to load from file the raster or the membrane potentials
        # saved by NEST
        try:
            raster = _readArray(tmpfile, sepchar=" ")
            raster = raster[:,1:3]
            raster[:,1] = raster[:,1]*get_time_step() # since dt might change, should really store the value of dt used
            for idx in xrange(len(raster)):
                result.write("%g\t%d\n" %(raster[idx][1], raster[idx][0]))
        except Exception:
            print "Error while writing data into a compatible mode"
        result.close()
        #os.system("rm %s" %tmpfile)
    else:
        shutil.move(tmpfile, filename)


def _print_v(tmpfile, filename, compatible_output=True):
    """ Print membrane potentials in a file, and postprocessed them if
    needed and asked to produce a compatible output for all the simulator
    Should actually work with record_v() and allow to dissociate the recording of the
    writing process, which is not the case for the moment"""
    pynest.sr('%s close' %tmpfile) 
    result = open(filename,'w',DEFAULT_BUFFER_SIZE)
    dt = get_time_step()
    n = int(get_current_time()/dt)
    result.write("# dt = %f\n# n = %d\n" % (dt, n))
    if (compatible_output):
        # Here we postprocess the file to have effectively the
        # desired format :
        # First line: dimensions of the population
        # Then spiketime cell_id-min(cell_id)

        # Pylab has a great load() function, but it is not necessary to import
        # it into pyNN. The fromfile() function of numpy has trouble on several
        # machine with Python 2.5, so that's why a dedicated _readArray function
        # has been created to load from file the raster or the membrane potentials
        # saved by NEST
        try:
            raster = _readArray(tmpfile, sepchar="\t")
            for idx in xrange(len(raster)):
                result.write("%g\t%d\n" %(raster[idx][1], raster[idx][0]))
        except Exception:
            print "Error while writing data into a compatible mode"
    else:
        f = open(tmpfile,'r',DEFAULT_BUFFER_SIZE)
        lines = f.readlines()
        f.close()
        for line in lines:
            result.write(line)
    result.close()
    os.system("rm %s" %tmpfile)

def _readArray(filename, sepchar = " ", skipchar = '#'):
    myfile = open(filename, "r", DEFAULT_BUFFER_SIZE)
    contents = myfile.readlines()
    myfile.close() 
    data = []
    for line in contents:
        stripped_line = line.lstrip()
        if (len(stripped_line) != 0):
            if (stripped_line[0] != skipchar):
                items = stripped_line.split(sepchar)
                # Here we have to deal with the fact that quite often, NEST
                # does not write correctly the last line of Vm recordings.
                # More precisely, it is often not complete
                try :
                    data.append(map(float, items))
                except Exception:
                    # The last line has a gid and just a "-" sign...
                    pass
    try :
        a = numpy.array(data)
    except Exception:
        # The last line has just a gid, so we has to remove it
        a = numpy.array(data[0:len(data)-2])
    (Nrow, Ncol) = a.shape
    if ((Nrow == 1) or (Ncol == 1)): a = ravel(a)
    return(a)

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
        
        common.Population.__init__(self, dims, cellclass, cellparams, label)  # move this to common.Population.__init__()
        
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
            id.parent = self
            #id.setCellClass(cellclass)
            #id.setPosition(self.locate(id))
            
        if self.cellparams:
            pynest.setDict(self.cell, self.cellparams)
            
        self.cell = numpy.reshape(self.cell, self.dim)    
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
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
        if len(addr) == self.ndim:
            id = self.cell[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim, str(addr))
        if addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        return id
    
    def __iter__(self):
        return self.cell.flat

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.__iter__():
            yield self.locate(i)
        
    def addresses(self):
        return self.__address_gen()
    
    def ids(self):
        return self.__iter__()
    
    def index(self, n):
        """Return the nth cell in the population (Indexing starts at 0)."""
        return self.cell.item(n)
    
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # The top two lines (commented out) are the original implementation,
        # which does not scale well when the population size gets large.
        # The next lines are the neuron implementation of the same method. This
        # assumes that the id values in self.cell are consecutive. This should
        # always be the case, I think? A unit test is needed to check this.
    
        ###assert isinstance(id, int)
        ###return tuple([a.tolist()[0] for a in numpy.where(self.cell == id)])
        
        id -= self.id_start
        if self.ndim == 3:
            rows = self.dim[1]; cols = self.dim[2]
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
    
    def get(self, parameter_name, as_array=False):
        """
        Get the values of a parameter for every cell in the population.
        """
        values = [getattr(cell, parameter_name) for cell in self.cell.flat]
        if as_array:
            values = numpy.array(values)
        return values
    
    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param, str):
            if isinstance(val, str) or isinstance(val, float) or isinstance(val, int):
                param_dict = {param:float(val)}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise common.InvalidParameterValueError
        for cell in self.cell.flat:
            cell.set_parameters(**param_dict)
        

    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        # Convert everything to 1D arrays
        cells = numpy.reshape(self.cell, self.cell.size)
        if self.cell.shape == value_array.shape: # the values are numbers or non-array objects
            values = numpy.reshape(value_array, self.cell.size)
        elif len(value_array.shape) == len(self.cell.shape)+1: # the values are themselves 1D arrays
            values = numpy.reshape(value_array, (self.cell.size, value_array.size/self.cell.size))
        else:
            raise common.InvalidDimensionsError, "Population: %s, value_array: %s" % (str(cells.shape),
                                                                                      str(value_array.shape))
        # Set the values for each cell
        if len(cells) == len(values):
            for cell,val in zip(cells, values):
                if not isinstance(val, str) and hasattr(val, "__len__"):
                    # tuples, arrays are all converted to lists, since this is
                    # what SpikeSourceArray expects. This is not very robust
                    # though - we might want to add things that do accept arrays.
                    val = list(val)
                if cell in self.cell.flat:
                    setattr(cell, parametername, val)
        else:
            raise common.InvalidDimensionsError
    
    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(rand_distr.rng, NativeRNG):
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            rarr = rand_distr.next(n=self.size)
            cells = numpy.reshape(self.cell, self.cell.size)
            assert len(rarr) == len(cells)
            for cell,val in zip(cells, rarr):
                setattr(cell, parametername, val)
        
    def _call(self, methodname, arguments):
        """
        Calls the method methodname(arguments) for every cell in the population.
        e.g. p.call("set_background","0.1") if the cell class has a method
        set_background().
        """
        raise Exception("Method not yet implemented")
    
    def _tcall(self, methodname, objarr):
        """
        `Topographic' call. Calls the method methodname() for every cell in the 
        population. The argument to the method depends on the coordinates of the
        cell. objarr is an array with the same dimensions as the Population.
        e.g. p.tcall("memb_init", vinitArray) calls
        p.cell[i][j].memb_init(vInitArray[i][j]) for all i,j.
        """
        raise Exception("Method not yet implemented")

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids
        of the cells to record._printSpikes(tmpfile, filename, compatible_output=True)
        """
        
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
        pynest.resCons(self.spike_detector[0], n_rec)

        if (fixed_list == True):
            for neuron in record_from:
                pynest.connect([neuron], self.spike_detector[0])
        else:
            for neuron in numpy.random.permutation(numpy.reshape(self.cell, (self.cell.size,)))[0:n_rec]:
                pynest.connect([neuron], self.spike_detector[0])
                
        # Open temporary output file & register file with detectors
        # This should be redone now that Eilif has implemented the pythondatum datum type
        # pynest.sr('/tmpfile_%s (tmpfile_%s) (w) file def' % (self.label, self.label)) # old
        file = "%s/%s.spikes" %(tempdir, self.label)
        pynest.sr('/%s.spikes (%s) (w) file def' %  (self.label, file))
        pynest.sr('%s << /output_stream %s.spikes >> SetStatus' % (pynest.getGID(self.spike_detector[0]), self.label))
        recorders['%s.spikes' %self.label] = ("spikes", file)
        self.n_rec = n_rec

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
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

        tmp_list = []
        filename    = '%s.v' % self.label
        record_file = tempdir+'/'+filename
        if (fixed_list == True):
            tmp_list = [[neuron] for neuron in record_from]
        else:
            for neuron in numpy.random.permutation(numpy.reshape(self.cell,(self.cell.size,)))[0:n_rec]:
                tmp_list.append([neuron])
        recorders['%s.v' %self.label] = ("v", record_file.replace('/','_'))
        pynest.record_v(tmp_list, record_file.replace('/','_'))
    
    def _get_tmp_file(self):
        file_label = '%s.spikes' % self.label
        (file_type, tmpfile) = recorders.pop(file_label)
        pynest.sr('%s close' % file_label)
        return tmpfile
    
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
        tmpfile = self._get_tmp_file()

        if (compatible_output):
            # Here we postprocess the file to have effectively the
            # desired format: spiketime (in ms) cell_id-min(cell_id)
            result = open(filename,'w',DEFAULT_BUFFER_SIZE)
            # Writing header:
            result.write("# " + "\t".join([str(d) for d in self.dim]) + "\n")
            result.write("# first_id = %d\n# last_id = %d\n" % (self.cell.flatten()[0], self.cell.flatten()[-1]))
            # Writing spiketimes, cell_id-min(cell_id)
            padding = numpy.reshape(self.cell, self.cell.size)[0]
            # Pylab has a great load() function, but it is not necessary to import
            # it into pyNN. The fromfile() function of numpy has trouble on several
            # machine with Python 2.5, so that's why a dedicated _readArray function
            # has been created to load from file the raster or the membrane potentials
            # saved by NEST
            try :
                raster = _readArray(tmpfile, sepchar=" ")
                #Sometimes, nest doesn't write the last line entirely, so we need
                #to trunk it to avoid errors
                raster = raster[:,1:3]
                raster[:,0] = raster[:,0] - padding
                raster[:,1] = raster[:,1]*get_time_step()
                for idx in xrange(len(raster)):
                    result.write("%g\t%d\n" %(raster[idx][1], raster[idx][0]))
            except Exception, e:
                print "Error while writing data into a compatible mode with file %s: %s" % (filename, e)
            result.close()
            os.system("rm %s" %tmpfile)
        else:
            print "didn't go into the compatible output stuff"
            shutil.move(tmpfile, filename)
    
    def getSpikes(self,  gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.
        """
        tmpfile = self._get_tmp_file()
        data = recording.readArray(tmpfile, sepchar=None)
        #data = _readArray(tmpfile, sepchar=" ")
        if data.size > 0:
            data = data[:,1:3]
            padding = self.cell.flatten()[0]
            data[:,0] -= padding
            data[:,1] *= get_time_step()
        return data

    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        status = pynest.get(pynest.getGID(self.spike_detector[0]))
        n_spikes = status["events"]
        return float(n_spikes)/self.n_rec

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init', rand_distr)
        #cells = numpy.reshape(self.cell, self.cell.size)
        #rvals = rand_distr.next(n=self.cell.size)
        #for node, v_init in zip(cells, rvals):
        #    pynest.setDict([node],{'u': v_init})
    
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

        file_label = '%s.v' % self.label
        (file_type, tmpfile) = recorders.pop(file_label)
        pynest.sr('%s close' % tmpfile)
        result = open(filename,'w',DEFAULT_BUFFER_SIZE)
        dt = get_time_step()
        n = int(get_current_time()/dt)
        result.write("# dt = %f\n# n = %d\n" % (dt, n))
        result.write("\n# first_id = %d\n# last_id = %d\n" % (self.cell.flatten()[0], self.cell.flatten()[-1]))
        if (compatible_output):
            result.write("# " + "\t".join([str(d) for d in self.dim]) + "\n")
            padding = numpy.reshape(self.cell, self.cell.size)[0]
            try:
                raster = _readArray(tmpfile, sepchar="\t")
                raster[:,0] = raster[:,0] - padding
                for idx in xrange(len(raster)):
                    result.write("%g\t%d\n" %(raster[idx][1], raster[idx][0]))
            except Exception, e:
                print "Error while writing data into a compatible mode with file %s: %s" % (filename, e)
        else:
            f = open(tmpfile,'r',DEFAULT_BUFFER_SIZE)
            lines = f.readlines()
            f.close()
            for line in lines:
                result.write(line)
        os.system("rm %s" %tmpfile)
        result.close()
     
    def describe(self):
        """
        Returns a human readable description of the population
        """
        print "\n------- Population description -------"
        print "Population called %s is made of %d cells" %(self.label, len(self.cell))
        print "-> Cells are aranged on a %dD grid of size %s" %(len(self.dim), self.dim)
        print "-> Celltype is %s" %self.celltype
        print "-> Cell Parameters used for cell[0] (during initialization and now) are: " 
        for key, value in self.cellparams.items():
          print "\t|", key, "\t: ", "init->", value, "\t now->", pynest.getDict([self.cell[0]])[0][key]
        print "--- End of Population description ----"
        
         

    
class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    class ConnectionDict:
            
            def __init__(self, parent):
                self.parent = parent
    
            def __getitem__(self, id):
                """Returns a (source address, target port number) tuple."""
                assert isinstance(id, int)
                return (pynest.getAddress(self.parent._sources[id]), self.parent._targetPorts[id])
    
    def __init__(self, presynaptic_population, postsynaptic_population, method='allToAll', method_parameters=None, source=None, target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentials
                 
        target - string specifying which synapse on the postsynaptic cell to
                 connect to
                 
        If source and/or target are not given, default values are used.
        
        method - string indicating which algorithm to use in determining
                 connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'.
        
        method_parameters - dict containing parameters needed by the connection
        method, although we should allow this to be a number or string if there
        is only one parameter.
        
        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.
        
        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method, method_parameters, source, target, synapse_dynamics, label, rng)
        
        self._targetPorts = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self._method  = method
        self._plasticity_model = "static_synapse"
        self.synapse_type = target
        
        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            self.nconn = connection_method(method_parameters)
        elif isinstance(method, common.Connector):
            self.nconn = method.connect(self)

        assert len(self._sources) == len(self._targets) == len(self._targetPorts), "Connection error. Source and target lists are of different lengths."
        self.connection = Projection.ConnectionDict(self)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        # This should already have been done if using a Connector object
        if isinstance(method, str) and (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(get_min_delay())
    
    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)
    
    def connections(self):
        """for conn in prj.connections()..."""
        for i in xrange(len(self)):
            yield self.connection[i]
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self, parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = AllToAllConnector(allow_self_connections)
        return c.connect(self)
    
    def _oneToOne(self, parameters=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        c = OneToOneConnector()
        return c.connect(self)

    def _fixedProbability(self, parameters):
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
        c = FixedProbabilityConnector(p_connect, allow_self_connections)
        return c.connect(self)
    
    def _distanceDependentProbability(self, parameters):
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
        c = DistanceDependentProbabilityConnector(d_expression, allow_self_connections=allow_self_connections)
        return c.connect(self)           
                
    def _fixedNumberPre(self, parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
            assert n > 0
            fixed = True
        elif type(parameters) == types.DictType:
            if parameters.has_key('n'): # all cells have same number of connections
                n = int(parameters['n'])
                assert n > 0
                fixed = True
            elif parameters.has_key('rand_distr'): # number of connections per cell follows a distribution
                rand_distr = parameters['rand_distr']
                assert isinstance(rand_distr, RandomDistribution)
                fixed = False
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        elif isinstance(parameters, RandomDistribution):
            rand_distr = parameters
            fixed = False
        else:
            raise Exception("Invalid argument type: should be an integer, dictionary or RandomDistribution object.")
         
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for pre in presynaptic_neurons:
            pre_addr = pynest.getAddress(pre)
            # Reserve space for connections
            if not fixed:
                n = rand_distr.next()
            pynest.resCons(pre_addr, n)                
            # pick n neurons at random
            for post in rng.permutation(postsynaptic_neurons)[0:n]:
                if allow_self_connections or (pre != post):
                    self._sources.append(pre)
                    self._targets.append(post)
                    self._targetPorts.append(pynest.connect(pre_addr, pynest.getAddress(post)))
    
    def _fixedNumberPost(self, parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        allow_self_connections = True
        if type(parameters) == types.IntType:
            n = parameters
            assert n > 0
            fixed = True
        elif type(parameters) == types.DictType:
            if parameters.has_key('n'): # all cells have same number of connections
                n = int(parameters['n'])
                assert n > 0
                fixed = True
            elif parameters.has_key('rand_distr'): # number of connections per cell follows a distribution
                rand_distr = parameters['rand_distr']
                assert isinstance(rand_distr, RandomDistribution)
                fixed = False
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        elif isinstance(parameters, RandomDistribution):
            rand_distr = parameters
            fixed = False
        else:
            raise Exception("Invalid argument type: should be an integer, dictionary or RandomDistribution object.")
         
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,))
        if self.rng:
            rng = self.rng
        else:
            rng = numpy.random
        for post in postsynaptic_neurons:
            post_addr = pynest.getAddress(post)
            # Reserve space for connections
            if not fixed:
                n = rand_distr.next()
            pynest.resCons(post_addr, n)                
            # pick n neurons at random
            for pre in rng.permutation(presynaptic_neurons)[0:n]:
                if allow_self_connections or (pre != post):
                    self._sources.append(pre)
                    self._targets.append(post)
                    self._targetPorts.append(pynest.connect(pynest.getAddress(pre), post_addr))
    
    def _fromFile(self, parameters):
        """
        Load connections from a file.
        """
        filename = parameters
        c = FromFileConnector(filename)
        return c.connect(self)
        
    def _fromList(self, conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        c = FromListConnector(conn_list)
        return c.connect(self)
        
    def _2D_Gauss(self, parameters):
        """
        Source neuron is connected to a 2D targetd population with a spatial profile (Gauss).
        parameters should have:
        rng:
        source_position: x,y of source neuron mapped to target populatio.
        source_id: source id
        n: number of synpases
        sigma: sigma of the Gauss
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
            r = rng.normal(scale=sigma, size=n)
            target_position_x = numpy.floor(pre_position[1]+r*numpy.cos(phi))
            target_position_y = numpy.floor(pre_position[0]+r*numpy.sin(phi))
            target_id = []
            for syn_nr in range(len(target_position_x)):
                #print syn_nr
                try:
                    # print target_position_x[syn_nr]
                    target_id.append(self.post[(target_position_x[syn_nr], target_position_y[syn_nr])])
                    # print target_id
                except IndexError:
                    target_id.append(False)
            
            pynest.divConnect(pre_id, target_id,[weight],[delay])
        
        
        n = parameters['n']
                
        if n > 0:
            ratio_dim_pre_post = ((1.*self.pre.dim[0])/(1.*self.post.dim[0]))
            print 'ratio_dim_pre_post', ratio_dim_pre_post
            run_id = 0

            for pre in numpy.reshape(self.pre.cell,(self.pre.cell.size)):
                #print 'pre', pre
                run_id +=1
                #print 'run_id', run_id
                if numpy.mod(run_id,500) == 0:
                    print 'run_id', run_id
                
                pre_position_tmp = self.pre.locate(pre)
                parameters['pre_position'] = numpy.divide(pre_position_tmp, ratio_dim_pre_post)
                parameters['pre_id'] = pre
                #a=Projection(self.pre, self.post,'rcf_2D', parameters)
                rcf_2D(parameters)
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        w = _convertWeight(w, self.synapse_type)
        if is_number(w):
           # set all the weights from a given node at once
            for src in numpy.reshape(self.pre.cell, self.pre.cell.size):
                assert isinstance(src, int), "GIDs should be integers"
                n = len(pynest.getDict([src])[0]['weights'])
                pynest.setDict([src], {'weights' : [w]*n})
        elif isinstance(w, list) or isinstance(w, numpy.ndarray):
            for src, port, weight in zip(self._sources, self._targetPorts, w):
                pynest.setWeight(pynest.getAddress(src), port, weight)
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        weights = _convertWeight(rand_distr.next(len(self)), self.synapse_type)
        for ((src, port), w) in zip(self.connections(), weights):
            pynest.setWeight(src, port, w)
    
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        if is_number(d):
            # Set all the delays from a given node at once.
            for src in numpy.reshape(self.pre.cell, self.pre.cell.size):
                assert isinstance(src, int), "GIDs should be integers"
                src_addr = pynest.getAddress(src)
                n = len(pynest.getDict([src_addr])[0]['delays'])
                pynest.setDict([src_addr], {'delays' : [d]*n})
        elif isinstance(d, list) or isinstance(d, numpy.ndarray):
            for src, port, delay in zip(self._sources, self._targetPorts, d):
                pynest.setDelay(pynest.getAddress(src), port, delay)
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        for src, port in self.connections():
            pynest.setDelay(src, port, rand_distr.next()[0])
    
    def setThreshold(self, threshold):
        """
        Where the emission of a spike is determined by watching for a
        threshold crossing, set the value of this threshold.
        """
        # This is a bit tricky, because in NEST the spike threshold is a
        # property of the cell model, whereas in NEURON it is a property of the
        # connection (NetCon).
        raise Exception("Method not yet implemented")
    
    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the synapse dynamics linked with the projection
        """
        raise Exception("Method not available ! Nest 1 does not support dynamical synapses")
    
    
    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        raise Exception("Method not available ! Nest 1 does not support dynamical synapses")
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        f = open(filename,'w',DEFAULT_BUFFER_SIZE)
        # Note unit change from pA to nA or nS to uS, depending on synapse type
        weights = [0.001*pynest.getWeight(src, port) for (src, port) in self.connections()]
        delays = [pynest.getDelay(src, port) for (src, port) in self.connections()] 
        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label,"%s", self.post.label,"%s","%g","%g")
        for i in xrange(len(self)):
            line = fmt  % (self.pre.locate(self._sources[i]),
                           self.post.locate(self._targets[i]),
                           weights[i],
                           delays[i])
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        file = open(filename,'w',DEFAULT_BUFFER_SIZE)
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,)).tolist()
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,)).tolist()
        weightArray = numpy.zeros((self.pre.size, self.post.size), dtype=float)
        for src in self._sources:
            src_addr = pynest.getAddress(src)
            pynest.sps(src_addr)
            pynest.sr('GetTargets')
            targetList = [pynest.getGID(tgt) for tgt in pynest.spp()]
            pynest.sps(src_addr)
            pynest.sr('GetWeights')
            weightList = pynest.spp()
            
            i = presynaptic_neurons.index(src)
            for tgt, w in zip(targetList, weightList):
                try:
                    j = postsynaptic_neurons.index(tgt)
                    weightArray[i][j] = 0.001*w
                except ValueError: # tgt is in a different population to the current postsynaptic population
                    pass
        fmt = "%g "*len(postsynaptic_neurons) + "\n"
        for i in xrange(weightArray.shape[0]):
            file.write(fmt % tuple(weightArray[i]))
        file.close()
            
    
    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        raise Exception("Method not yet implemented")
    
    def describe(self):
        """
        Return a human readable description of the projection
        """
        print "\n------- Projection description -------"
        print "Projection %s from %s [%d cells] to %s [%d cells]" %(self.label, self.pre.label, len(self.pre.cell),self.post.label, len(self.post.cell))
        print "Connector used is %s : " %self._method
        if isinstance(self._method.weights,RandomDistribution):
          print "\t| Weights are drawn from %s distribution with parameters %s "%(self._method.weights.name, self._method.weights.parameters)
        else:
          print "\t| Weights: ", self._method.weights
        if isinstance(self._method.delays,RandomDistribution):
          print "\t| Delays are drawn from %s distribution with parameters %s " %(self._method.delays.name, self._method.delays.parameters)
        else:
          print "\t| Delays: ", self._method.delays
        print "\t| Plasticity: ", self._plasticity_model
        print "\t --> %d connections have been created for this projection" %len(self)
        print "To check, here are the parameters of one connection from this projection"
        print "\tsource\ttarget"
        print "\t%d\t%d" %(self._sources[0], self._targets[0])
        print "\t| Weight: ", pynest.getWeight([self._sources[0]],self._targetPorts[0])
        print "\t| Delay: ", pynest.getDelay([self._sources[0]], self._targetPorts[0])
        
        print "---- End of Projection description -----"

# ==============================================================================
#   Utility classes
# ==============================================================================
   
Timer = common.Timer

# ==============================================================================
