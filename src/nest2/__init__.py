# -*- coding: utf-8 -*-
"""
PyNEST v2 implementation of the PyNN API.
$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev:188 $"

import nest
from pyNN import common
from pyNN.random import *
import numpy, types, sys, shutil, os, logging, copy, tempfile
from math import *
from pyNN.nest2.cells import *
from pyNN.nest2.connectors import *


recorder_dict  = {}
tempdirs       = []
dt             = 0.1
_min_delay     = 0.1
recording_device_names = {'spikes': 'spike_detector',
                          'v': 'voltmeter',
                          'conductance': 'conductancemeter'}


# ==============================================================================
#   Utility classes and functions
# ==============================================================================

class ID(common.ID):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """
    
    def __getattr__(self,name):
        """ """
        nest_parameters = nest.GetStatus([int(self)])[0]
        if issubclass(self.cellclass, common.StandardCellType):
            #translated_name = self.cellclass.translations[name][0]
            #pname = self.cellclass.translations[name]['translated_name']
            pval = eval(self.cellclass.translations[name]['reverse_transform'], {}, nest_parameters)
        elif isinstance(self.cellclass, str) or self.cellclass is None:
            #translated_name = name
            pval = nest_parameters[name]
        else:
            raise Exception("ID object has invalid cell class %s" % str(self.cellclass))
        #return nest.GetStatus([int(self)])[0][translated_name]
        return pval
    
    def setParameters(self,**parameters):
        # We perform a call to the low-level function set() of the API.
        # If the cellclass is not defined in the ID object :
        #if (self.cellclass == None):
        #    raise Exception("Unknown cellclass")
        #else:
        #    # We use the one given by the user
        set(self, self.cellclass, parameters) 

    def getParameters(self):
        """ """
        nest_parameters = nest.GetStatus([int(self)])[0]
        pynn_parameters = {}
        if issubclass(self.cellclass, common.StandardCellType):
            for k,v in self.cellclass.translations.items():
                pynn_parameters[k] = eval(self.cellclass.translations[k]['reverse_transform'], {}, nest_parameters)
        return pynn_parameters
            


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
            delay = _min_delay
        return delay
    
    def convertWeight(self, w, synapse_type):
        
        if isinstance(w, RandomDistribution):
            weight = RandomDistribution(w.name, w.parameters, w.rng)
            if weight.name == "uniform":
                (w_min,w_max) = weight.parameters
                weight.parameters = (1000.*w_min, 1000.*w_max)
            elif weight.name ==  "normal":
                (w_mean,w_std) = weight.parameters
                weight.parameters = (1000.*w_mean, w_std*1000.)
        else:
            weight = w*1000.

        if synapse_type == 'inhibitory':
            # We have to deal with the distribution, and anticipate the
            # fact that we will need to multiply by a factor 1000 the weights
            # in nest...
            if isinstance(weight, RandomDistribution):
                if weight.name == "uniform":
                    print weight.name, weight.parameters
                    (w_min,w_max) = weight.parameters
                    if w_min >= 0 and w_max >= 0:
                        weight.parameters = (-w_max, -w_min)
                elif weight.name ==  "normal":
                    (w_mean,w_std) = weight.parameters
                    if w_mean > 0:
                        weight.parameters = (-w_mean, w_std)
                else:
                    print "WARNING: no conversion of the inhibitory weights for this particular distribution"
            elif weight > 0:
                weight *= -1
        return weight


class Connection(object):
    
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post
        try:
            conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
        except Exception:
            raise common.ConnectionError
        if (len(conn_dict['targets']) == 0):
            raise common.ConnectionError
        if conn_dict:
            self.port = len(conn_dict['targets'])-1
        else:
            raise Exception("Could not get port number for connection between %s and %s" % (pre, post))

    def _set_weight(self, w):
        pass

    def _get_weight(self):
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
        conn_dict = nest.GetConnections([self.pre],'static_synapse')[0]
        if conn_dict:
            return conn_dict['weights'][self.port]
        else:
            return None
    
    def _set_delay(self, d):
        pass

    def _get_delay(self):
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
        conn_dict = nest.GetConnections([self.pre],'static_synapse')[0]
        if conn_dict:
            return conn_dict['delays'][self.port]
        else:
            return None
        
    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

def list_standard_models():
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1,min_delay=0.1,max_delay=0.1,debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    #if min_delay > max_delay:
    #    raise Exception("min_delay has to be less than or equal to max_delay.")
    global dt
    global tempdir
    global _min_delay
    global _max_delay
    #global hl_spike_files, hl_v_files
    dt = timestep
    _min_delay = min_delay
    _max_delay = max_delay
    
    # reset the simulation kernel
    nest.ResetKernel()
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')

#    # check if hl_spike_files , hl_v_files are empty
#    if not len(hl_spike_files) == 0:
#        print 'hl_spike_files still contained files, please close all open files before setup'
#        hl_spike_files = {}
#    if not len(hl_v_files) == 0:
#        print 'hl_v_files still contained files, please close all open files before setup'
#        hl_v_files = {}
    
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    
    # set tempdir
    nest.SetStatus([0], {'device_prefix':tempdir})
    # set resolution
    nest.SetStatus([0], {'resolution': dt})
    
    # Should be corrected and fixed. Should be done for all synapse type
    #nest.SetSynapseDefaults('static_synapse',{'delay': min_delay, 'min_delay': min_delay,
    #'max_delay' : max_delay})
    #nest.SetSynapseDefaults('static_synapse',{'delay': min_delay})
    
    if extra_params.has_key('threads'):
        if extra_params.has_key('kernelseeds'):
            print 'params has kernelseeds ', extra_params['kernelseeds']
            kernelseeds = extra_params['kernelseeds']
        else:
            # default kernelseeds, for each thread one, to ensure same for each sim we get the rng with seed 42
            rng = NumpyRNG(42)
            num_processes = nest.GetStatus([0])[0]['num_processes']
            kernelseeds = (rng.rng.uniform(size=extra_params['threads']*num_processes)*100).astype('int').tolist()
            print 'params has no kernelseeds, we use ',kernelseeds
            
        nest.SetStatus([0],[{'local_num_threads' : extra_params['threads'],
                             'rng_seeds'         : kernelseeds}])

    
    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning()
    if debug:
        if isinstance(debug, basestring):
            filename = debug
        else:
            filename = "nest.log"
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=filename,
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='nest.log',
                    filemode='w')

    logging.info("Initialization of Nest")    
    return nest.Rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    # We close the high level files opened by populations objects
    # that may have not been written.

    # NEST will soon close all its output files after the simulate function is over, therefore this step is not necessary
    global tempdirs
    
    # And we postprocess the low level files opened by record()
    # and record_v() method
    #for file in ll_spike_files:
    #    _printSpikes(file, compatible_output)
    #for file, nest_file in ll_v_files:
    #    _print_v(file, nest_file, compatible_output)
    print "Saving the following files:", recorder_dict.keys()
    for filename in recorder_dict.keys():
        _print(filename, gather=False, compatible_output=compatible_output)
    
    for tempdir in tempdirs:
        os.system("rm -rf %s" %tempdir)

def run(simtime):
    """Run the simulation for simtime ms."""
    nest.Simulate(simtime)

def setRNGseeds(seedList):
    """Globally set rng seeds."""
    nest.SetStatus([0],{'rng_seeds': seedList})

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
        cell_gids = nest.Create(celltype.nest_name,n)
        cell_gids = [ID(gid) for gid in cell_gids]
        nest.SetStatus(cell_gids, [celltype.parameters])
    elif isinstance(cellclass, str):  # celltype is not a standard cell
        cell_gids = nest.Create(cellclass,n)
        cell_gids = [ID(gid) for gid in cell_gids]
        if paramDict:
            nest.SetStatus(cell_gids, [paramDict])
    else:
        raise "Invalid cell type"
    for id in cell_gids:
    #    #id.setCellClass(cellclass)
        id.cellclass = cellclass
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def connect(source,target,weight=None,delay=None,synapse_type=None,p=1,rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or uS."""
    global dt
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = _min_delay
    # If the delay is too small , we have to throw an error
    if delay < _min_delay or delay > _max_delay:
        raise common.ConnectionError
    weight = weight*1000 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                         # Using convention in this way is not ideal. We should be able to look up the units used by each model somewhere.
    if synapse_type == 'inhibitory' and weight > 0:
        weight *= -1
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            nest.ConnectWD([source],[target],[weight],[delay])
            connect_id = Connection(source, target)
        else:
            connect_id = []
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            for src in source:
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = rng.rng.uniform(0,1,len(target))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0,1,len(target))
                for j,tgt in enumerate(target):
                    if p >= 1 or rarr[j] < p:
                        nest.ConnectWD([src],[tgt],[weight],[delay])
                        connect_id += [Connection(src,tgt)]
    #except nest.SLIError:
    except Exception: # unfortunately, SLIError seems to have disappeared.Hopefully it will be reinstated.
        raise common.ConnectionError
    return connect_id

def set(cells,cellclass,param,val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    cellclass must be supplied for doing translation of parameter names."""
    # we should just assume that cellclass has been defined and raise an Exception if it has not
    if val:
        param = {param:val}
    try:
        i = cells[0]
    except TypeError:
        cells = [cells]
    if not (isinstance(cellclass,str) or cellclass is None):
        if issubclass(cellclass, common.StandardCellType):
            param = cellclass({}).translate1(param)
        else:
            raise TypeError, "cellclass must be a string, None, or derived from commonStandardCellType"
    nest.SetStatus(cells,[param])

def _connect_recording_device(recorder, record_from=None):
    #print "Connecting recorder %s to cell(s) %s" % (recorder, record_from)
    device = nest.GetStatus(recorder, "model")[0]
    if device == "spike_detector":
        nest.ConvergentConnect(record_from, recorder)
    elif device in ('voltmeter', 'conductancemeter'):        
        nest.DivergentConnect(recorder, record_from)
    else:
        raise Exception("Not a valid recording device")

def _record(variable, source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    device_name = recording_device_names[variable]
    
    recording_device = nest.Create(device_name)
    nest.SetStatus(recording_device,
                   {"to_file" : True, "withgid" : True, "withtime" : True,
                    "interval": nest.GetStatus([0], "resolution")[0]})
    print "Trying to record %s from cell %s using %s %s (filename=%s)" % (variable, source, device_name, recording_device, filename)
            
    if type(source) != types.ListType:
        source = [source]
    _connect_recording_device(recording_device, record_from=source)
    if filename is not None:
        recorder_dict[filename] = recording_device 

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    _record('spikes', source, filename) 
    
def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    _record('v', source, filename) 

def _print(user_filename, gather=True, compatible_output=True, population=None, variable=None):
    global recorder_dict
    
    if population is None:
        recorder = recorder_dict[user_filename]
    else:
        assert variable in ['spikes', 'v', 'conductance']
        recorder = population.recorders[variable]
    
    #print "Printing to %s from recorder %s (compatible_output=%s)" % (user_filename, recorder, compatible_output)
    
    nest.FlushDevice(recorder) 
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])
    
    # First combine data from different threads
    os.system("rm -f %s" % user_filename+'_%d'%nest.Rank())
    for nest_thread in range(local_num_threads):
        nest.sps(recorder[0])
        nest.sr("%i GetAddress %i append" % (recorder[0], nest_thread))
        nest.sr("GetStatus /filename get")
        nest_filename = nest.spp() #nest.GetStatus(recorder, "filename")
        merged_filename = "%s/%s_%d" % (os.path.dirname(nest_filename), user_filename, nest.Rank())
        system_line = 'cat %s >> %s' % (nest_filename, merged_filename) 
        #print system_line
        os.system(system_line)
        os.remove(nest_filename)
    if gather and len(node_list) > 1:
        nest.sps(recorder[0])
        nest.sr("GetStatus /filename get")
        nest_filename = nest.spp() #nest.GetStatus(recorder, "filename")
        if nest.Rank() == 0: # only on the master node (?)
            for node in node_list:
                merged_filename = "%s/%s_%d" % (os.path.dirname(nest_filename), user_filename, node)
                system_line = 'cat %s >> %s' % (nest_filename, merged_filename)
                #print system_line
                os.system(system_line)
                os.remove(nest_filename)
   
    if compatible_output:
        if gather == False or nest.Rank() == 0: # if we gather, only do this on the master node
            logging.info("Writing %s in compatible format." % user_filename)
            
            # Here we postprocess the file to have effectively the
            # desired format: spiketime (in ms) cell_id-min(cell_id)
            #if not os.path.exists(user_filename):
            result = open(user_filename,'w',10000)
            #else:
            #    result = open(user_filename,'a',1000)
                
            ## Writing header info (e.g., dimensions of the population)
            if population is not None:
                result.write("# dimensions =" + "\t".join([str(d) for d in population.dim]) + "\n")
                result.write("# first_id = %d\n" % population.cell[0])
                result.write("# last_id = %d\n" % population.cell[-1])
                padding = population.cell.flatten()[0]
            else:
                padding = 0
            result.write("# dt = %g\n" % dt)
                            
            # Writing spiketimes, cell_id-min(cell_id)
                 
            # (Pylab has a great load() function, but it is not necessary to import
            # it into pyNN. The fromfile() function of numpy has trouble on several
            # machine with Python 2.5, so that's why a dedicated _readArray function
            # has been created to load from file the raster or the membrane potentials
            # saved by NEST).
                    
            # open file
            if int(os.path.getsize(merged_filename)) > 0:
                data = _readArray(merged_filename, sepchar=None)
                data[:,0] = data[:,0] - padding
                # sort
                #indx = data.argsort(axis=0, kind='mergesort')[:,0] # will quicksort (not stable) work?
                #data = data[indx]
                if data.shape[1] == 4: # conductance files
                    raise Exception("Not yet implemented")
                elif data.shape[1] == 3: # voltage files
                    result.write("# n = %d\n" % int(nest.GetStatus([0], "time")[0]/dt))
                    for idx in xrange(len(data)):
                        result.write("%g\t%d\n" % (data[idx][2], data[idx][0])) # v id
                elif data.shape[1] == 2: # spike files
                    for idx in xrange(len(data)):
                        result.write("%g\t%d\n" % (data[idx][1], data[idx][0])) # time id
                else:
                    raise Exception("Data file should have 2,3 or 4 columns, actually has %d" % data.shape[1])
            else:
                logging.info("%s is empty" % merged_filename)
            result.close()

    if population is None:
        recorder_dict.pop(user_filename)    

def _get(population=None, variable=None):
    global recorder_dict

    if population is None:
        recorder = recorder_dict[user_filename]
    else:
        assert variable in ['spikes', 'v', 'conductance']
        recorder = population.recorders[variable]

    #print "Printing to %s from recorder %s (compatible_output=%s)" % (user_filename, recorder, compatible_output)

    nest.FlushDevice(recorder)
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])

    # Combine data from different threads to the zeroeth thread
    nest.sps(recorder[0])
    nest.sr("%i GetAddress %i append" % (recorder[0], 0))
    nest.sr("GetStatus /filename get")
    base_filename = nest.spp() #nest.GetStatus(recorder, "filename")

    if local_num_threads>1:
        for nest_thread in range(1,local_num_threads):
            nest.sps(recorder[0])
            nest.sr("%i GetAddress %i append" % (recorder[0], nest_thread))
            nest.sr("GetStatus /filename get")
            nest_filename = nest.spp() #nest.GetStatus(recorder, "filename")
            system_line = 'cat %s >> %s' % (nest_filename, base_filename)
            os.system(system_line)
            os.remove(nest_filename)

    # now we have the merged_filename

    if population is not None:
        padding = population.cell.flatten()[0]
    else:
        padding = 0


    data = _readArray(base_filename, sepchar=None)
    data[:,0] = data[:,0] - padding
    os.remove(base_filename)

    return data


def _readArray(filename, sepchar=None, skipchar='#'):
    logging.debug(filename)
    myfile = open(filename, "r")
    contents = myfile.readlines()
    myfile.close()
    logging.debug(contents)
    data = []
    for line in contents:
        stripped_line = line.lstrip()
        if (len(stripped_line) != 0):
            if (stripped_line[0] != skipchar):
                items = stripped_line.split(sepchar)
                #if len(items) != 3:
                #    print stripped_line
                #    print items
                #    raise Exception()
                data.append(map(float, items))
    #try :
    a = numpy.array(data)
    #except Exception:
    #    raise
        # The last line has just a gid, so we has to remove it
        #a = numpy.array(data[0:len(data)-2])
    (Nrow,Ncol) = a.shape
    logging.debug(str(a.shape))
    if ((Nrow == 1) or (Ncol == 1)): a = numpy.ravel(a)
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
            self.cell = nest.Create(self.celltype.nest_name, self.size)
            self.cellparams = self.celltype.parameters
        elif isinstance(cellclass, str):
            self.cell = nest.Create(cellclass, self.size)
            
        self.cell_local = numpy.array(self.cell)[numpy.array(nest.GetStatus(self.cell,'local'))]
        
        self.cell = numpy.array([ ID(GID) for GID in self.cell ], ID)
        self.id_start = self.cell.reshape(self.size,)[0]
        
        for id in self.cell:
            id.parent = self
            #id.setCellClass(cellclass)
            #id.setPosition(self.locate(id))
            
        if self.cellparams:
            nest.SetStatus(self.cell_local, [self.cellparams])
            
        self.cell = numpy.reshape(self.cell, self.dim)    
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {'spikes': None, 'v': None, 'conductance': None}
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
    
        ###assert isinstance(id,int)
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
    
    def index(self, n):
        """Return the nth cell in the population."""
        if hasattr(n, '__len__'):
            n = numpy.array(n)
        return self.cell[n]
    
    def set(self,param,val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
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
            paramDict = self.celltype.translate1(paramDict)
        nest.SetStatus(self.cell_local, [paramDict])

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
            parametername = self.celltype.translate1({parametername: values[0]}).keys()[0]
        # Set the values for each cell
        if len(cells) == len(values):
            for cell,val in zip(cells,values):
                try:
                    if not isinstance(val,str) and hasattr(val,"__len__"):
                        val = list(val) # tuples, arrays are all converted to lists, since this is what SpikeSourceArray expects. This is not very robust though - we might want to add things that do accept arrays.
                    else:
                        if cell in self.cell_local:
                            nest.SetStatus([cell],[{parametername: val}])
                #except nest.SLIError:
                except Exception: # unfortunately, SLIError seems to have disappeared.
                    raise common.InvalidParameterValueError, "Error from SLI"
        else:
            raise common.InvalidDimensionsError
        
    
    def rset(self,parametername,rand_distr):
        """
        'Random' set. Sets the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(self.celltype, common.StandardCellType):
            parametername = self.celltype.translate1({parametername: 0.0}).keys()[0]
        if isinstance(rand_distr.rng, NativeRNG):
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            #cells = numpy.reshape(self.cell,self.cell.size)
            #rarr = rand_distr.next(n=self.size)
            rarr = rand_distr.next(n=len(self.cell_local))
            cells = self.cell_local
            assert len(rarr) == len(cells)
            for cell,val in zip(cells,rarr):
                try:
                    nest.SetStatus([cell],{parametername: val})
                #except nest.SLIError:
                except Exception: # unfortunately, SLIError seems to have disappeared.
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

    def _record(self, variable, record_from=None, rng=None):
        assert variable in ('spikes', 'v', 'conductance')
    
        # create device
        device_name = recording_device_names[variable]
        if self.recorders[variable] is None:
            self.recorders[variable] = nest.Create(device_name)
            nest.SetStatus(self.recorders[variable],
                           {"to_file" : True, "withgid" : True, "withtime" : True,
                            "interval": nest.GetStatus([0], "resolution")[0]})      
        
        # create list of neurons        
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
            
        if variable == 'spikes':
            self.n_rec = n_rec
        
        tmp_list = []
        if (fixed_list == True):
            for neuron in record_from:
                tmp_list = [neuron for neuron in record_from]
        else:
            # should use `rng` here, if provided
            for neuron in numpy.random.permutation(numpy.reshape(self.cell,(self.cell.size,)))[0:n_rec]:
                tmp_list.append(neuron)
                
        # connect device to neurons
        _connect_recording_device(self.recorders[variable], record_from=tmp_list)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids
        of the cells to record.
        """
        self._record('spikes', record_from, rng)
        
    def record_v(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('v', record_from, rng)
    
    def record_c(self,record_from=None,rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('conductance', record_from, rng)
    
    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Writes spike times to file.
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster.
        If gather is True, the file will only be created on the master node,
        otherwise, a file will be written on each node.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="spikes")
    
    def getSpikes(self):
        """
        Returns a numpy array of the spikes of the population

        Useful for small populations, for example for single neuron Monte-Carlo.

        NOTE: getSpikes or printSpikes should be called only once per run,
        because they mangle simulator recorder files.
        """
        return _get(population=self, variable="spikes")
    
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        n_spikes = nest.GetStatus(self.recorders['spikes'], "events")[0]
        return float(n_spikes)/self.n_rec

    def randomInit(self, rand_distr):
        """
        Sets initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init',rand_distr)

    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        If compatible_output is True, the format is "t v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="v")
               
    def print_c(self,filename,gather=True, compatible_output=True):
        """
        Write conductance traces to file.
        If compatible_output is True, the format is "t g cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="conductance")

    
class Projection(common.Projection, WDManager):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """    
    class ConnectionDict:
            
            def __init__(self,parent):
                self.parent = parent
    
            def __getitem__(self,id):
                """Returns a (source address,target port number) tuple."""
                assert isinstance(id, int)
                return (self.parent._sources[id], self.parent._targetPorts[id])
    
    def __init__(self,presynaptic_population,postsynaptic_population,method='allToAll',
                 methodParameters=None,source=None,target=None,synapse_dynamics=None,label=None,rng=None):
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
        common.Projection.__init__(self,presynaptic_population,postsynaptic_population,
                                   method,methodParameters,source,target,synapse_dynamics,label,rng)
        
        self._targetPorts = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self.synapse_type = target
        
        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            self.nconn = connection_method(methodParameters)
        elif isinstance(method,common.Connector):
            self.nconn = method.connect(self)

        #assert len(self._sources) == len(self._targets) == len(self._targetPorts), "Connection error. Source and target lists are of different lengths."
        self.connection = Projection.ConnectionDict(self)
    
    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)
    
    def connections(self):
        """for conn in prj.connections()..."""
        for i in xrange(len(self)):
            yield self.connection[i]
    
    # --- Connection methods ---------------------------------------------------
    
    def _allToAll(self,parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = AllToAllConnector(allow_self_connections)
        return c.connect(self)
    
    def _oneToOne(self,parameters=None):
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

    def _fixedProbability(self,parameters):
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
    
    def _distanceDependentProbability(self,parameters):
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
                
    def _fixedNumberPre(self,parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPreConnector(n, allow_self_connections)
        return c.connect(self)
    
    def _fixedNumberPost(self,parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPostConnector(n, allow_self_connections)
        return c.connect(self)
    
    def _fromFile(self,parameters):
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
            f = open(filename,'r',10000)
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise "Argument type not yet implemented"
        
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            src=eval(src)
            tgt=eval(tgt)
            input_tuples.append((src,tgt,float(w),float(d)))
        f.close()
        
        self._fromList(input_tuples)
        
    def _fromList(self,conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = eval("self.pre%s" %src)
            tgt = eval("self.post%s" %tgt)
            pre_addr = nest.GetAddress([src])
            post_addr = nest.GetAddress([tgt])
            nest.ConnectWD(pre_addr,post_addr, [1000*weight], [delay])
            self._sources.append(src)
            self._targets.append(tgt)
            self._targetPorts.append(tgt)

            

    def _test_delay(self,params):
        # debug get delays from outside
        #delay_array = parameters['delays_array']
        #weight_array = parameters['weights_array']
        #target_id = parameters['target_id']
        #pre_id = parameters['pre_id']
        print 'inside test_delay'
        print 'delays ',params['delays_array']
        print 'weights ',params['weights_array']
        print 'pre_id ',params['pre_id']
        print 'target_id ',params['target_id']
        eval(params['eval_string'])
        #cons=nest.divConnect(params['pre_id'],params['target_id'],params['weights_array'].tolist(),params['delays_array'].tolist())
        #nest.divConnect(pre_id,target_id,weight_array.tolist(),delay_array.tolist())
        print 'leaving test_delay'
        


    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        w = self.convertWeight(w, self.synapse_type)
        if type(w) == types.FloatType or type(w) == types.IntType or type(w) == numpy.float64 :
            # set all the weights from a given node at once
            for src in self.pre.cell.flat:
                conn_dict = nest.GetConnections([src], 'static_synapse')[0]
                if conn_dict:
                    n = len(conn_dict['weights'])
                nest.SetConnections([src], 'static_synapse', [{'weights': [w]*n}])
        elif isinstance(w,list) or isinstance(w,numpy.ndarray):
            raise Exception("Not yet implemented")
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeWeights(self,rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        rand_distr = self.convertWeight(rand_distr, self.synapse_type)
        for src in self.pre.cell.flat:
            conn_dict = nest.GetConnections([src], 'static_synapse')[0]
            n = len(conn_dict['weights'])
            weights = rand_distr.next(n).tolist()
            # if self.synapse_type == 'inhibitory', should we *= -1 ???
            nest.SetConnections([src], 'static_synapse', [{'weights': weights}])
    
    def setDelays(self,d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        if type(d) == types.FloatType or type(d) == types.IntType or type(d) == numpy.float64:
            d = float(d)
            # set all the weights from a given node at once
            for src in self.pre.cell.flat:
                conn_dict = nest.GetConnections([src], 'static_synapse')[0]
                if conn_dict:
                    n = len(conn_dict['delays'])
                nest.SetConnections([src], 'static_synapse', [{'delays': [d]*n}])
        elif isinstance(d,list) or isinstance(d,numpy.ndarray):
            raise Exception("Not yet implemented")
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
    
    def randomizeDelays(self,rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        for src in self.pre.cell.flat:
            conn_dict = nest.GetConnections([src], 'static_synapse')[0]
            n = len(conn_dict['delays'])
            delays = rand_distr.next(n).tolist()
            nest.SetConnections([src], 'static_synapse', [{'delays': delays}])
    
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
        f = open(filename,'w',10000)
        # Note unit change from pA to nA or nS to uS, depending on synapse type
        def getWD(wd, src, port):
            conn_dict = nest.GetConnection([src],'static_synapse',port)
            if isinstance(conn_dict, dict):
                return conn_dict[wd]
            else:
                raise Exception("Either the source id (%s) or the port number (%s) or both is invalid." % (src, port))
        #weights = [0.001*nest.GetConnection([src],'static_synapse',port)['weight'] for (src,port) in self.connections()]
        #delays = [nest.GetConnection([src],'static_synapse',port)['delay'] for (src,port) in self.connections()]
        weights = [0.001*getWD('weight',src,port) for (src,port) in self.connections()]
        delays = [getWD('delay',src,port) for (src,port) in self.connections()]
        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label,"%s",self.post.label,"%s","%g","%g")
        for i in xrange(len(self)):
            line = fmt  % (self.pre.locate(self._sources[i]),
                           self.post.locate(self._targets[i]),
                           weights[i],
                           delays[i])
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()
    
    def printWeights(self,filename,format=None,gather=True):
        """Print synaptic weights to file."""
        file = open(filename,'w',10000)
        postsynaptic_neurons = numpy.reshape(self.post.cell,(self.post.cell.size,)).tolist()
        presynaptic_neurons  = numpy.reshape(self.pre.cell,(self.pre.cell.size,)).tolist()
        weightArray = numpy.zeros((self.pre.size,self.post.size),dtype=float)
        for src in self._sources:
            src_addr = nest.getAddress(src)
            nest.sps(src_addr)
            nest.sr('GetTargets')
            targetList = [nest.getGID(tgt) for tgt in nest.spp()]
            nest.sps(src_addr)
            nest.sr('GetWeights')
            weightList = nest.spp()
            
            i = presynaptic_neurons.index(src)
            for tgt,w in zip(targetList,weightList):
                try:
                    j = postsynaptic_neurons.index(tgt)
                    weightArray[i][j] = w
                except ValueError: # tgt is in a different population to the current postsynaptic population
                    pass
        fmt = "%g "*len(postsynaptic_neurons) + "\n"
        for i in xrange(weightArray.shape[0]):
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
