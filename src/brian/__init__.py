# -*- coding: utf-8 -*-
"""
PyNEST implementation of the PyNN API.
$Id:nest1.py 143 2007-10-05 14:20:16Z apdavison $
"""
__version__ = "$Rev: 294 $"

import brian_no_units_no_warnings
import brian
from pyNN import common, recording
from pyNN.random import *
import numpy, types, sys, shutil, os, logging, copy, tempfile
from math import *
from pyNN.brian.cells import *
from pyNN.brian.connectors import *
from pyNN.brian.synapses import *

net      = None
simclock = None
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
    
    def __getattr__(self, name):
        try:
            val = float(self.parent.brian_cells[int(self)].__getattr__(name))
        except KeyError:
            raise NonExistentParameterError(name, self.cellclass)
        return val
    
    def set_native_parameters(self, parameters):
        for key, value in parameters.items():
            self.parent.brian_cells[int(self)].__setattr__(key,value)
        

        
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
    global tempdir, net, simclock
    
    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning() 
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='brian.log',
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='brian.log',
                    filemode='w')

    logging.info("Initialization of Brian")
    timestep  = 0.001*timestep
    min_delay = 0.001*min_delay
    max_delay = 0.001*max_delay
    net       = brian.Network()
    simclock  = brian.Clock(dt=timestep)
    return 0

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""

def run(simtime):
    """Run the simulation for simtime ms."""
    global net
    # The run() command of brian accept second
    net.run(0.001*simtime)

def get_min_delay():
    return 0.1*brian.ms
common.get_min_delay = get_min_delay

def get_time_step():
    return 0.1*brian.ms
common.get_time_step = get_time_step

def get_current_time():
    pass

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
    pass

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    pass

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value."""
    # we should just assume that cellclass has been defined and raise an Exception if it has not
    pass

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    pass


def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    pass


def _printSpikes(tmpfile, filename, compatible_output=True):
    """ Print spikes into a file, and postprocessed them if
    needed and asked to produce a compatible output for all the simulator
    Should actually work with record() and allow to dissociate the recording of the
    writing process, which is not the case for the moment"""
    pass


def _print_v(tmpfile, filename, compatible_output=True):
    """ Print membrane potentials in a file, and postprocessed them if
    needed and asked to produce a compatible output for all the simulator
    Should actually work with record_v() and allow to dissociate the recording of the
    writing process, which is not the case for the moment"""
    pass

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
        global net, simclock
        common.Population.__init__(self, dims, cellclass, cellparams, label)  # move this to common.Population.__init__()
        
        # Should perhaps use "LayoutNetwork"?
        
        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cellparams = self.celltype.parameters
            if isinstance(self.celltype,SpikeSourcePoisson):
                rate       = self.cellparams['rate']
                fct        = self.celltype.fct
                self.brian_cells  = brian.PoissonGroup(self.size, rates = fct, clock=simclock)
            else:
                v_thresh   = self.cellparams['v_thresh']
                v_reset    = self.cellparams['v_reset']
                tau_refrac = self.cellparams['tau_refrac']
                self.brian_cells = brian.NeuronGroup(self.size,model=cellclass.eqs,threshold=v_thresh,reset=v_reset, refractory=tau_refrac, clock=simclock, compile=True)

        elif isinstance(cellclass, str):
            v_thresh   = self.cellparams['v_thresh']
            v_reset    = self.cellparams['v_reset']
            tau_refrac = self.cellparams['tau_refrac']
            self.brian_cells = brian.NeuronGroup(self.size,model=cellclass,threshold=v_thresh,reset=v_reset, clock=simclock)
            self.cellparams = self.celltype.parameters

        useless_params=['v_thresh','v_reset','tau_refrac','cm']
        if self.cellparams:
            for key, value in self.cellparams.items():
                if not key in useless_params:
                    setattr(self.brian_cells,key,value)
        self.cell = numpy.array([ID(cell) for cell in xrange(len(self.brian_cells))],ID)
        for id in self.cell:
            id.parent = self
        self.cell = numpy.reshape(self.cell, self.dim)
        self.spike_recorder = None
        self.vm_recorder    = None
        self.ce_recorder    = None
        self.ci_recorder    = None
        self.first_id       = 0
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        Population.nPop += 1
        net.add(self.brian_cells)
    
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
        values = getattr(self.brian_cells, parameter_name)
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
        for key, value in para_dict.items():
            setattr(self.brian_cells, key, value)
        

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
            assert len(rarr) == len(self.brian_cells)
            if parametername in self.celltype.scaled_parameters():
                translation = self.celltype.translations[parametername]
                rarr = eval(translation['forward_transform'], globals(), {parametername : rarr})
                setattr(self.brian_cells, translation['translated_name'], rarr)
            elif parametername in self.celltype.simple_parameters():
                translation = self.celltype.translations[parametername]
                setattr(self.brian_cells, translation['translated_name'], rarr)
            else:
                for cell,val in zip(self.cell.flat, rarr):
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
        population. The argument     to the method depends on the coordinates of the
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
        global net
        if record_from:
            if isinstance(record_from,list):
                N = len(record_from)
            if isinstance(record_from,int):
                N = record_from
            print "Warning: Brian can record only the %d first cells of the population" %N
            self.spike_recorder = brian.SpikeMonitor(self.brian_cells[0:N],True)
        else:
            self.spike_recorder = brian.SpikeMonitor(self.brian_cells,True)
        net.add(self.spike_recorder)

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        global net
        if record_from:
            if isinstance(record_from,int):
                N = record_from
                record_from = numpy.random.permutation(self.cell.flatten()[0:N])
            self.vm_recorder = brian.StateMonitor(self.brian_cells,'v',record=record_from)
        else:
            self.vm_recorder = brian.StateMonitor(self.brian_cells,'v',record=True)
        net.add(self.vm_recorder)
    
    def record_c(self, record_from=None, rng=None, to_file=True):
        """
        If record_from is not given, record the conductances/currents for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        global net
        if record_from:
            if isinstance(record_from,int):
                N = record_from
                record_from = numpy.random.permutation(self.cell.flatten()[0:N])
            self.ce_recorder = brian.StateMonitor(self.brian_cells,'ge',record=record_from)
            self.ci_recorder = brian.StateMonitor(self.brian_cells,'gi',record=record_from)
        else:
            self.ce_recorder = brian.StateMonitor(self.brian_cells,'ge',record=True)
            self.ci_recorder = brian.StateMonitor(self.brian_cells,'gi',record=True)
        net.add(self.ce_recorder)
        net.add(self.ci_recorder)
    
    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Write spike times to file.
        
        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        dt = get_time_step()*1000
        if self.spike_recorder:
            f = open(filename,"w", DEFAULT_BUFFER_SIZE)
            f.write("# dimensions =" + "\t".join([str(d) for d in self.dim]) + "\n")
            f.write("# first_id = %d\n" % self.first_id)
            f.write("# last_id = %d\n" % (self.first_id+len(self)-1,))
            f.write("# dt = %g\n" % dt)
            spikes = numpy.array(self.spike_recorder.spikes)
            spikes[:,1]=1000*spikes[:,1]
            for item in spikes:
                f.write("%g\t%d\n" %(item[1], item[0]))
            f.close()
    
    def getSpikes(self,  gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.
        """
        return self.spike_recorder.spikes

    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        
        # TO DO : add a recordede list otherwise wrong
        
        return float(self.spike_recorder.nspikes)/len(self)

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init', rand_distr)


    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.
        
        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep and number of data points per cell is written as a header,
        indicated by a '#' at the beginning of the line.
        
        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        
        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        dt = get_time_step()*1000
        if self.vm_recorder:
            f = open(filename,"w", DEFAULT_BUFFER_SIZE)
            N = len(self.vm_recorder[0])
            f.write("# dimensions =" + "\t".join([str(d) for d in self.dim]) + "\n")
            f.write("# first_id = %d\n" % self.first_id)
            f.write("# last_id = %d\n" % (self.first_id+len(self)-1,))
            f.write("# dt = %g\n" % dt)
            f.write("# n = %d\n" % N)
            cells = self.vm_recorder.get_record_indices()
            for cell in cells:
                vm = 1000*self.vm_recorder[cell]
                for idx in xrange(N):
                    f.write("%g\t%g\n" %(vm[idx],cell))
            f.close()

    
class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    
    def __init__(self, presynaptic_population, postsynaptic_population, method='allToAll', method_parameters=None, source=None, target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.
        
        source - string specifying which attribute of the presynaptic cell
                 signals action potentialss
                 
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
        global net
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method, method_parameters, source, target, synapse_dynamics, label, rng)
        
        self._method  = method
        self._connections = None
        self._plasticity_model = "static_synapse"
        self.synapse_type = target
        
        if isinstance(method, str):
            connection_method = getattr(self,'_%s' % method)   
            self.nconn = connection_method(method_parameters)
        elif isinstance(method, common.Connector):
            self.nconn = method.connect(self)
            net.add(self._connections)
        
        # By defaut, we set all the delays to min_delay, except if
        # the Projection data have been loaded from a file or a list.
        # This should already have been done if using a Connector object
        if isinstance(method, str) and (method != 'fromList') and (method != 'fromFile'):
            self.setDelays(get_min_delay())
    
    def __len__(self):
        """Return the total number of connections."""
        return self._connections.W.getnnz()
    
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
        
    
    # --- Methods for setting connection parameters ----------------------------
    
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        raise Exception("With Brian, weights should be specified in the connector object and can not be changed afterwards !")
    
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        raise Exception("With Brian, weights should be specified in the connector object and can not be changed afterwards !")
    
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        raise Exception("With Brian, delays should be specified in the connector object and can not be changed afterwards !")
    
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        raise Exception("Method not available ! Brian does not support non homogeneous delays!")
    
    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the synapse dynamics linked with the projection
        """
        raise Exception("Method not available ! Brian does not support dynamical synapses")
    
    
    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        raise Exception("Method not available ! Brian does not support dynamical synapses")
    
    # --- Methods for writing/reading information to/from file. ----------------
    
    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        pass
    
    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        pass
            
    
    def weightHistogram(self, min=None, max=None, nbins=10):
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
