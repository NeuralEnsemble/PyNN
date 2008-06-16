# encoding: utf-8
"""
nrnpython implementation of the PyNN API.
                                                      
$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev: 191 $"

from pyNN.random import *
from pyNN.neuron2.utility import *
from pyNN import common, utility
from pyNN.neuron2.cells import *
from pyNN.neuron2.connectors import *
from pyNN.neuron2.synapses import *

from math import *
import sys
import numpy
import logging
import operator

# Global variables
gid_counter = 0
initialised = False
running = False
quit_on_end = True
recorder_list = []

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False,**extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global initialised, quit_on_end, running, pc
    if not initialised:
        h('min_delay = 0')
        h('tstop = 0')
        pc = neuron.ParallelContext()
        pc.spike_compress(1,0)
        cvode = neuron.CVode()
        utility.init_logging("neuron2.log.%d" % rank(), debug)
        logging.info("Initialization of NEURON (use setup(.., debug=True) to see a full logfile)")
    h.dt = timestep
    h.tstop = 0
    h.min_delay = min_delay
    running = False
    if 'quit_on_end' in extra_params:
        quit_on_end = extra_params['quit_on_end']
    if extra_params.has_key('use_cvode'):
        cvode.active(int(extra_params['use_cvode']))
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)
    pc.runworker()
    pc.done()
    if quit_on_end:
        logging.info("Finishing up with NEURON.")
        h.quit()
        
def run(simtime):
    """Run the simulation for simtime ms."""
    global running
    logging.info("Running the simulation for %d ms" % simtime)
    if not running:
        running = True
        local_minimum_delay = pc.set_maxstep(10)
        h.finitialize()
        h.tstop = 0
        logging.debug("local_minimum_delay on host #%d = %g" % (rank(), local_minimum_delay))
        if num_processes() > 1:
            assert local_minimum_delay >= get_min_delay(),\
                   "There are connections with delays (%g) shorter than the minimum delay (%g)" % (local_minimum_delay, get_min_delay())
    h.tstop = simtime
    pc.psolve(h.tstop)
    return get_current_time()

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

def get_current_time():
    """Return the current time in the simulation."""
    return h.t

def get_time_step():
    return h.dt
common.get_time_step = get_time_step

def get_min_delay():
    return h.min_delay
common.get_min_delay = get_min_delay

def num_processes():
    return int(pc.nhost())

def rank():
    """Return the MPI rank."""
    return int(pc.id())

def list_standard_models():
    return [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]

class ID(int, common.IDMixin):
    
    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def _build_cell(self, celltype, parent=None):
        gid = int(self)
        self._cell = celltype.model(**celltype.parameters)  # create the cell object
        pc.set_gid2node(gid, rank())                        # assign the gid to this node
        nc = neuron.NetCon(self._cell.source, None)         # } associate the cell spike source
        pc.cell(gid, nc.hoc_obj)                            # } with the gid (using a temporary NetCon)
        self.parent = parent
    
    def get_native_parameters(self):
        D = {}
        for name in ('tau_m', 'cm', 'v_rest', 'v_thresh', 't_refrac',
                     'i_offset', 'v_reset', 'v_init', 'tau_e', 'tau_i'):
            D[name] = getattr(self._cell, name)
        return D
    
    def set_native_parameters(self, parameters):
        for name, val in parameters.items():
            setattr(self._cell, name, val)
            

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

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
        global gid_counter
        common.Population.__init__(self, dims, cellclass, cellparams, label)
        self.recorders = {'spikes': Recorder('spikes', population=self),
                          'v': Recorder('v', population=self)}
        self.first_id = gid_counter
        self.last_id = gid_counter + self.size
        self.label = self.label or 'population%d' % Population.nPop
        
        # Build the arrays of cell ids
        # Cells on the local node are represented as ID objects, other cells by integers
        # All are stored in a single numpy array for easy lookup by address
        # The local cells are also stored in a list, for easy iteration
        self._all_ids = numpy.array([id for id in range(self.first_id, self.last_id)], ID) #.reshape(self.dim)
        # _mask_local is used to extract those elements from arrays that apply to the cells on the current node, e.g. for tset()
        self._mask_local = self._all_ids%num_processes()==0 # round-robin distribution of cells between nodes
        ## _map_local is used to map addresses to the index within the list of local cells
        #self._map_local = numpy.where(self._mask_local, self._all_ids/int(num_processes()), None)
        for i,(id,is_local) in enumerate(zip(self._all_ids, self._mask_local)):
            if is_local:
                self._all_ids[i] = ID(id)
        # _local_ids is a list containing only those cells that exist on the current node
        self._local_ids = self._all_ids[self._mask_local]
        self._all_ids = self._all_ids.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        
        self.celltype = cellclass(cellparams)
        for id in self._local_ids:
            id._build_cell(self.celltype, parent=self)
        gid_counter += self.size
        Population.nPop += 1
        logging.info(self.describe('Creating Population "%(label)s" of shape %(dim)s, '+
                                   'containing `%(celltype)s`s with indices between %(first_id)s and %(last_id)s'))
        logging.debug(self.describe())
    
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
            id = self._all_ids[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim, str(addr))
        if addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        return id
    
    def __iter__(self):
        """Iterator over cell ids."""
        return iter(self._local_ids)

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.__iter__():
            yield self.locate(i)

    def addresses(self):
        """Iterator over cell addresses."""
        return self.__address_gen()

    def ids(self):
        """Iterator over cell ids."""
        return self.__iter__()
    
    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        id = id - self.first_id
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
        values = [getattr(cell, parameter_name) for cell in self]
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
            if isinstance(val, (str, float, int)):
                param_dict = {param: float(val)}
            elif isinstance(val, (list, numpy.ndarray)):
                param_dict = {param: val}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise common.InvalidParameterValueError
        logging.info("%s.set(%s)", self.label, param_dict)
        for cell in self:
            cell.set_parameters(**param_dict)
    
    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        if self.dim == value_array.shape: # the values are numbers or non-array objects
            local_values = value_array[self._mask_local]
        elif len(value_array.shape) == len(self.dim)+1: # the values are themselves 1D arrays
            #values = numpy.reshape(value_array, (self.dim, value_array.size/self.size))
            local_values = value_array[self._mask_local] # not sure this works
        else:
            raise common.InvalidDimensionsError, "Population: %s, value_array: %s" % (str(self.dim),
                                                                                      str(value_array.shape))
        ##local_indices = numpy.array([cell.gid for cell in self]) - self.first_id
        ##values = values.take(local_indices) # take just the values for cells on this machine
        assert local_values.size == self._local_ids.size, "%d != %d" % (local_values.size, self._local_ids.size)
        
        logging.info("%s.tset('%s', array(shape=%s, min=%s, max=%s)",
                     self.label, parametername, value_array.shape,
                     value_array.min(), value_array.max())
        # Set the values for each cell
        for cell, val in zip(self, local_values.flat):
            setattr(cell, parametername, val)

    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        # Note that we generate enough random numbers for all cells on all nodes
        # but use only those relevant to this node. This ensures that the
        # sequence of random numbers does not depend on the number of nodes,
        # provided that the same rng with the same seed is used on each node.
        if isinstance(rand_distr.rng, NativeRNG):
            rng = h.Random(rand_distr.rng.seed or 0)
            native_rand_distr = getattr(rng, rand_distr.name)
            rarr = [native_rand_distr(*rand_distr.parameters)] + [rng.repick() for i in range(self._all_ids.size-1)]
        else:
            rarr = rand_distr.next(n=self._all_ids.size)
        rarr = numpy.array(rarr)
        logging.info("%s.rset('%s', %s)", self.label, parametername, rand_distr)
        for cell,val in zip(self, rarr[self._mask_local.flatten()]):
            setattr(cell, parametername, val)

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init', rand_distr)

    def __record(self, record_what, record_from=None, rng=None):
        """
        Private method called by record() and record_v().
        """
        global myid
        fixed_list=False
        if isinstance(record_from, list): #record from the fixed list specified by user
            fixed_list=True
        elif record_from is None: # record from all cells:
            record_from = self._all_ids
        elif isinstance(record_from, int): # record from a number of cells, selected at random  
            # Each node will record N/nhost cells...
            nrec = int(record_from/num_processes())
            if not rng:
                rng = numpy.random
            record_from = rng.permutation(self._all_ids)[0:nrec]
            # need ID objects, permutation returns integers
            # ???
        else:
            raise Exception("record_from must be either a list of cells or the number of cells to record from")
        # record_from is now a list or numpy array. We do not have to worry about whether the cells are
        # local because the Recorder object takes care of this.
        self.recorders[record_what].record(record_from)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self.__record('spikes', record_from, rng)

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self.__record('v', record_from, rng)

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
        self.recorders['spikes'].write(filename, gather, compatible_output)

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
        self.recorders['v'].write(filename, gather, compatible_output)

    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.
        """
        return self.recorders['spikes'].get(gather)

    def get_v(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.
        """
        return self.recorders['v'].get(gather)

    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        n_spikes = len(self.recorders['spikes'].get(gather))
        n_rec = len(self.recorders['spikes'].recorded)
        return float(n_spikes)/n_rec

    def describe(self, template=None):
        if template is None:
            rows = ['==== Population %(label)s ====',
                    'Dimensions: %(dim)s',
                    'Cell type: %(celltype)s',
                    'ID range: %(first_id)d-%(last_id)d',
                    'First cell on this node:',
                    '  ID: %(local_first_id)d',
                    '  Parameters: %(cell_parameters)s',
                   ]
            template = "\n".join(rows)
        context = self.__dict__.copy()
        first_id = self._local_ids[0]
        context.update(local_first_id=first_id)
        context.update(cell_parameters=first_id.get_parameters())
        context.update(celltype=self.celltype.__class__.__name__)
        return template % context