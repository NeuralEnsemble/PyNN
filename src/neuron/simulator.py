"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    create_cells()
    reset()
    run()
    finalize()

Classes:
    ID
    Recorder
    ConnectionManager
    Connection
    
Attributes:
    state -- a singleton instance of the _State class.
    recorder_list

All other functions and classes are private, and should not be used by other
modules.
    
$Id$
"""

from pyNN import __path__ as pyNN_path
from pyNN import common, recording, random
import platform
import logging
import numpy
import os.path
from neuron import h

# Global variables
nrn_dll_loaded = []
recorder_list = []
connection_managers = []


# --- Internal NEURON functionality -------------------------------------------- 

def load_mechanisms(path=pyNN_path[0]):
    # this now exists in the NEURON distribution, so could probably be removed
    global nrn_dll_loaded
    if path not in nrn_dll_loaded:
        arch_list = [platform.machine(), 'i686', 'x86_64', 'powerpc']
        # in case NEURON is assuming a different architecture to Python, we try multiple possibilities
        for arch in arch_list:
            lib_path = os.path.join(path, 'hoc', arch, '.libs', 'libnrnmech.so')
            if os.path.exists(lib_path):
                h.nrn_load_dll(lib_path)
                nrn_dll_loaded.append(path)
                return
        raise Exception("NEURON mechanisms not found in %s." % os.path.join(path, 'hoc'))

def is_point_process(obj):
    """Determine whether a particular object is a NEURON point process."""
    return hasattr(obj, 'loc')

def register_gid(gid, source, section=None):
    """Register a global ID with the global `ParallelContext` instance."""
    state.parallel_context.set_gid2node(gid, state.mpi_rank)  # assign the gid to this node
    if is_point_process(source):
        nc = h.NetCon(source, None)                          # } associate the cell spike source
    else:
        nc = h.NetCon(source, None, sec=section)
    state.parallel_context.cell(gid, nc)              # } with the gid (using a temporary NetCon)

def nativeRNG_pick(n, rng, distribution='uniform', parameters=[0,1]):
    """
    Pick random numbers from a Hoc Random object.
    
    Return a Numpy array.
    """
    native_rng = h.Random(0 or rng.seed)
    rarr = [getattr(native_rng, distribution)(*parameters)]
    rarr.extend([native_rng.repick() for j in xrange(n-1)])
    return numpy.array(rarr)

def h_property(name):
    """Return a property that accesses a global variable in Hoc."""
    def _get(self):
        return getattr(h,name)
    def _set(self, val):
        setattr(h, name, val)
    return property(fget=_get, fset=_set)

class _Initializer(object):
    """
    Manage initialization of NEURON cells. Rather than create an
    `FInializeHandler` instance for each cell that needs to initialize itself,
    we create a single instance, and use an instance of this class to maintain
    a list of cells that need to be initialized.
    
    Public methods:
        register()
    """
    
    def __init__(self):
        """
        Create an `FinitializeHandler` object in Hoc, which will call the
        `_initialize()` method when NEURON is initialized.
        """
        self.cell_list = []
        self.population_list = []
        h('objref initializer')
        h.initializer = self
        self.fih = h.FInitializeHandler("initializer._initialize()")
    
    #def __call__(self):
    #    """This is to make the Initializer a Singleton."""
    #    return self
    
    def register(self, *items):
        """
        Add items to the list of cells/populations to be initialized. Cell
        objects must have a `memb_init()` method.
        """
        for item in items:
            if isinstance(item, common.Population):
                if "Source" not in item.celltype.__class__.__name__: # don't do memb_init() on spike sources
                    self.population_list.append(item)
            else:
                if hasattr(item._cell, "memb_init"):
                    self.cell_list.append(item)
    
    def _initialize(self):
        """Call `memb_init()` for all registered cell objects."""
        logging.info("Initializing membrane potential of %d cells and %d Populations." % \
                     (len(self.cell_list), len(self.population_list)))
        for cell in self.cell_list:
            cell._cell.memb_init()
        for population in self.population_list:
            for cell in population:
                cell._cell.memb_init()


# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.gid_counter = 0
        self.running = False
        self.initialized = False
        h('min_delay = 0')
        h('tstop = 0')
        self.parallel_context = h.ParallelContext()
        self.parallel_context.spike_compress(1,0)
        self.num_processes = int(self.parallel_context.nhost())
        self.mpi_rank = int(self.parallel_context.id())
        self.cvode = h.CVode()
        self.max_delay = 1e12
        h('objref plastic_connections')
        h.plastic_connections = []
    
    t = h_property('t')
    dt = h_property('dt')
    tstop = h_property('tstop')         # } do these really need to be stored in hoc?
    min_delay = h_property('min_delay') # }


def reset():
    """Reset the state of the current network to time t = 0."""
    state.running = False
    state.t = 0
    state.tstop = 0

def run(simtime):
    """Advance the simulation for a certain time."""
    if not state.running:
        state.running = True
        local_minimum_delay = state.parallel_context.set_maxstep(10)
        h.finitialize()
        state.tstop = 0
        logging.debug("local_minimum_delay on host #%d = %g" % (state.mpi_rank, local_minimum_delay))
        if state.num_processes > 1:
            assert local_minimum_delay >= state.min_delay,\
                   "There are connections with delays (%g) shorter than the minimum delay (%g)" % (local_minimum_delay, state.min_delay)
    state.tstop += simtime
    logging.info("Running the simulation for %d ms" % simtime)
    state.parallel_context.psolve(state.tstop)

def finalize(quit=True):
    """Finish using NEURON."""
    state.parallel_context.runworker()
    state.parallel_context.done()
    if quit:
        logging.info("Finishing up with NEURON.")
        h.quit()


# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__
    
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def _build_cell(self, cell_model, cell_parameters):
        """
        Create a cell in NEURON, and register its global ID.
        
        `cell_model` -- one of the cell classes defined in the
                        `neuron.cells` module (more generally, any class that
                        implements a certain interface, but I haven't
                        explicitly described that yet).
        `cell_parameters` -- a dictionary containing the parameters used to
                             initialise the cell model.
        """
        gid = int(self)
        self._cell = cell_model(**cell_parameters)          # create the cell object
        register_gid(gid, self._cell.source, section=self._cell) # not adequate for multi-compartmental cells
        
    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEURON cell model."""
        D = {}
        for name in self._cell.parameter_names:
            D[name] = getattr(self._cell, name)
        return D
    
    def set_native_parameters(self, parameters):
        """Set parameters of the NEURON cell model from a dictionary."""
        for name, val in parameters.items():
            setattr(self._cell, name, val)


# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    numpy1_1_formats = {'spikes': "%g\t%d",
                        'v': "%g\t%g\t%d",
                        'gsyn': "%g\t%g\t%g\t%d"}
    numpy1_0_formats = {'spikes': "%g", # only later versions of numpy support different
                        'v': "%g",      # formats for different columns
                        'gsyn': "%g"}
    formats = {'spikes': 'id t',
               'v': 'id t v',
               'gsyn': 'id t ge gi'}
    
    def __init__(self, variable, population=None, file=None):
        """
        Create a recorder.
        
        `variable` -- "spikes", "v" or "gsyn"
        `population` -- the Population instance which is being recorded by the
                        recorder (optional)
        `file` -- one of:
            - a file-name,
            - `None` (write to a temporary file)
            - `False` (write to memory).
        """
        self.variable = variable
        self.filename = file or None
        self.population = population # needed for writing header information
        self.recorded = set([])
        
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        logging.debug('Recorder.record(%s)', str(ids))
        if self.population:
            ids = set([id for id in ids if id in self.population.local_cells])
        else:
            ids = set([id for id in ids if id.local])
        new_ids = list( ids.difference(self.recorded) )
        
        self.recorded = self.recorded.union(ids)
        logging.debug('Recorder.recorded = %s' % self.recorded)
        if self.variable == 'spikes':
            for id in new_ids:
                id._cell.record(1)
        elif self.variable == 'v':
            for id in new_ids:
                id._cell.record_v(1)
        elif self.variable == 'gsyn':
            for id in new_ids:
                id._cell.record_gsyn("excitatory", 1)
                id._cell.record_gsyn("inhibitory", 1)
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
        
    def get(self, gather=False, compatible_output=True, offset=None):
        """Return the recorded data as a Numpy array."""
        # compatible_output is not used, but is needed for compatibility with the nest2 module.
        # Does nest2 really need it?
        if offset is None:
            if self.population:
                offset = self.population.first_id
            else:
                offset = 0
                
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.recorded:
                spikes = numpy.array(id._cell.spike_times)
                spikes = spikes[spikes<=state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape)*(id-offset), spikes]).T
                    data = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.recorded:
                v = numpy.array(id._cell.vtrace)  
                t = numpy.array(id._cell.record_times)
                #new_data = numpy.array([t, v, numpy.ones(v.shape)*id]).T
                #new_data = numpy.array([numpy.ones(v.shape)*id, t, v]).T                
                new_data = numpy.array([numpy.ones(v.shape)*(id-offset), t, v]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            data = numpy.empty((0,4))
            for id in self.recorded:
                ge = numpy.array(id._cell.gsyn_trace['excitatory'])
                gi = numpy.array(id._cell.gsyn_trace['inhibitory'])
                t = numpy.array(id._cell.record_times)             
                new_data = numpy.array([numpy.ones(ge.shape)*(id-offset), t, ge, gi]).T
                data = numpy.concatenate((data, new_data))
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
        if gather and state.num_processes > 1:
            data = recording.gather(data)
        return data
    
    def write(self, file=None, gather=False, compatible_output=True):
        """Write recorded data to file."""
        data = self.get(gather)
        filename = file or self.filename
        #recording.rename_existing(filename) # commented out because it causes problems when running with mpirun and a shared filesystem. Probably a timing problem
        try:
            numpy.savetxt(filename, data, Recorder.numpy1_0_formats[self.variable], delimiter='\t')
        except AttributeError, errmsg:
            # we assume the error is due to the lack of savetxt in older versions of numpy and
            # so provide a cut-down version of that function
            f = open(filename, 'w')
            fmt = Recorder.numpy1_0_formats[self.variable]
            for row in data:
                f.write('\t'.join([fmt%val for val in row]) + '\n')
            f.close()
        if compatible_output and state.mpi_rank==0: # would be better to distribute this step
            recording.write_compatible_output(filename, filename, self.variable,
                                              Recorder.formats[self.variable],
                                              self.population, state.dt)
        
    def count(self, gather=False):
        """
        Return the number of data points for each cell, as a dict. This is mainly
        useful for spike counts or for variable-time-step integration methods.
        """
        N = {}
        if self.variable == 'spikes':
            for id in self.recorded:
                N[id] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes.")
        if gather and state.num_processes > 1:
            N = recording.gather_dict(N)
        return N
        

# --- For implementation of create() and Population.__init__() -----------------

def create_cells(cellclass, cellparams, n, parent=None):
    """
    Create cells in NEURON.
    
    `cellclass`  -- a PyNN standard cell or a native NEURON cell class that
                   implements an as-yet-undescribed interface.
    `cellparams` -- a dictionary of cell parameters.
    `n`          -- the number of cells to create
    `parent`     -- the parent Population, or None if the cells don't belong to
                    a Population.
    
    This function is used by both `create()` and `Population.__init__()`
    
    Return:
        - a 1D array of all cell IDs
        - a 1D boolean array indicating which IDs are present on the local MPI
          node
        - the ID of the first cell created
        - the ID of the last cell created
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, basestring): # cell defined in hoc template
        try:
            cell_model = getattr(h, cellclass)
        except AttributeError:
            raise common.InvalidModelError("There is no hoc template called %s" % cellclass)
        cell_parameters = cellparams or {}
    elif isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
        celltype = cellclass(cellparams)
        cell_model = celltype.model
        cell_parameters = celltype.parameters
    else:
        cell_model = cellclass
        cell_parameters = cellparams
    first_id = state.gid_counter
    last_id = state.gid_counter + n - 1
    all_ids = numpy.array([id for id in range(first_id, last_id+1)], ID)
    # mask_local is used to extract those elements from arrays that apply to the cells on the current node
    mask_local = all_ids%state.num_processes==state.mpi_rank # round-robin distribution of cells between nodes
    for i,(id,is_local) in enumerate(zip(all_ids, mask_local)):
        all_ids[i] = ID(id)
        all_ids[i].parent = parent
        if is_local:
            all_ids[i].local = True
            all_ids[i]._build_cell(cell_model, cell_parameters)
        else:
            all_ids[i].local = False
    initializer.register(*all_ids[mask_local])
    state.gid_counter += n
    return all_ids, mask_local, first_id, last_id


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, source, target, nc):
        """
        Create a new connection.
        
        `source` -- ID of pre-synaptic neuron.
        `target` -- ID of post-synaptic neuron.
        `nc` -- a Hoc NetCon object.
        """
        self.source = source
        self.target = target
        self.nc = nc
        
    def useSTDP(self, mechanism, parameters, ddf):
        """
        Set this connection to use spike-timing-dependent plasticity.
        
        `mechanism`  -- the name of an NMODL mechanism that modifies synaptic
                        weights based on the times of pre- and post-synaptic spikes.
        `parameters` -- a dictionary containing the parameters of the weight-
                        adjuster mechanism.
        `ddf`        -- dendritic delay fraction. If ddf=1, the synaptic delay
                        `d` is considered to occur entirely in the post-synaptic
                        dendrite, i.e., the weight adjuster receives the pre-
                        synaptic spike at the time of emission, and the post-
                        synaptic spike a time `d` after emission. If ddf=0, the
                        synaptic delay is considered to occur entirely in the
                        pre-synaptic axon.
        """
        self.ddf = ddf
        self.weight_adjuster = getattr(h, mechanism)(0.5)
        self.pre2wa = state.parallel_context.gid_connect(int(self.source), self.weight_adjuster)
        self.pre2wa.threshold = self.nc.threshold
        self.pre2wa.delay = self.nc.delay * (1-ddf)
        self.pre2wa.weight[0] = 1
        # directly create NetCon as wa is on the same machine as the post-synaptic cell
        self.post2wa = h.NetCon(self.target._cell.source, self.weight_adjuster)
        self.post2wa.threshold = 1
        self.post2wa.delay = self.nc.delay * ddf
        self.post2wa.weight[0] = -1
        for name, value in parameters.items():
            setattr(self.weight_adjuster, name, value)
        # setpointer
        i = len(h.plastic_connections)
        h.plastic_connections.append(self)
        h('setpointer plastic_connections._[%d].weight_adjuster.wsyn, plastic_connections._[%d].nc.weight' % (i,i))

    def _set_weight(self, w):
        self.nc.weight[0] = w

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return self.nc.weight[0]

    def _set_delay(self, d):
        self.nc.delay = d
        if hasattr(self, 'pre2wa'):
            self.pre2wa.delay = float(d)*(1-self.ddf)
            self.post2wa.delay = float(d)*self.ddf

    def _get_delay(self):
        """Connection delay in ms."""
        return self.nc.delay

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)


class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_model` -- not used. Present for consistency with other simulators.
        `parent` -- the parent `Projection`, if any.
        """
        global connection_managers
        self.connections = []
        self.parent = parent
        connection_managers.append(self)
        
    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        return self.connections[i]
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(self.connections)
    
    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        return iter(self.connections)
    
    def connect(self, source, targets, weights, delays, synapse_type):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        `synapse_type` -- a string identifying the synapse to connect to. Should
                          be "excitatory", "inhibitory", or the name of any
                          other attribute of the post-synaptic cell that refers
                          to a synaptic mechanism. May be `None`, which is
                          treated the same as "excitatory".
        """
        if not isinstance(source, int) or source > state.gid_counter or source < 0:
            errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, state.gid_counter)
            raise common.ConnectionError(errmsg)
        if not common.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise common.ConnectionError("Invalid target ID: %s" % target)
              
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if synapse_type is None:
                    synapse_type = weight>=0 and 'excitatory' or 'inhibitory' 
                synapse_object = getattr(target._cell, synapse_type)
                nc = state.parallel_context.gid_connect(int(source), synapse_object)
                nc.weight[0] = weight
                nc.delay  = delay
                self.connections.append(Connection(source, target, nc))

    def get(self, parameter_name, format, offset=(0,0)):
        """
        Get the values of a given attribute (weight, delay, etc) for all
        connections on the local MPI node.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        `offset` -- an (i,j) tuple giving the offset to be used in converting
                    source and target IDs to array indices.
        
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
            values = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            for c in self.connections:
                value = getattr(c, parameter_name)
                addr = (c.source-offset[0], c.target-offset[1])
                if numpy.isnan(values[addr]):
                    values[addr] = value
                else:
                    values[addr] += value
        else:
            raise Exception("format must be 'list' or 'array'")
        return values

    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections.
        """
        if common.is_number(value):
            for c in self:
                setattr(c, name, value)
        elif common.is_listlike(value):
            for c,val in zip(self.connections, value):
                setattr(c, name, val)
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")


# --- Initialization, and module attributes ------------------------------------

load_mechanisms() # maintains a list of mechanisms that have already been imported
state = _State()  # a Singleton, so only a single instance ever exists
del _State
initializer = _Initializer()
del _Initializer