# encoding: utf-8
import logging
import brian
import numpy
from itertools import izip
import scipy.sparse
from pyNN import common, cells, recording

mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz

# Global variables
recorder_list = []
ZERO_WEIGHT = 1e-99

def _new_property(obj_hierarchy, attr_name, units):
    """
    Returns a new property, mapping attr_name to obj_hierarchy.attr_name.
    
    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """
    def set(self, value):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value*units)
    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        return getattr(obj, attr_name)/units
    return property(fset=set, fget=get)


class ThresholdNeuronGroup(brian.NeuronGroup):
    
    def __init__(self, n, equations):
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   threshold=-50.0*mV,
                                   reset=-60.0*mV,
                                   refractory=100.0*ms,
                                   compile=True,
                                   clock=state.simclock,
                                   max_delay=state.max_delay*ms,
                                   )
        self.v_init = -60.0*mV
        self.parameter_names = equations._namespace.keys() + ['v_thresh', 'v_reset', 'tau_refrac', 'v_init']
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        
    tau_refrac = _new_property('_resetfun', 'period', ms)
    v_reset = _new_property('_resetfun', 'resetvalue', mV)
    v_thresh = _new_property('_threshold', 'threshold', mV)
    
    def _set_v_init(self, v_init):
        self._S0[self.var_index['v']] = v_init
        self.v = v_init
    def _get_v_init(self):
        return self._S0[self.var_index['v']]
        
    v_init = property(fget=_get_v_init, fset=_set_v_init)

        
class PoissonGroupWithDelays(brian.PoissonGroup):

    def __init__(self, N, rates=0):
        brian.NeuronGroup.__init__(self, N, model=brian.LazyStateUpdater(),
                                   threshold=brian.PoissonThreshold(),
                                   clock=state.simclock,
                                   max_delay=state.max_delay*ms)
        if callable(rates): # a function is passed
            self._variable_rate=True
            self.rates=rates
            self._S0[0]=self.rates(self.clock.t)
        else:
            self._variable_rate=False
            self._S[0,:]=rates
            self._S0[0]=rates
        #self.var_index={'rate':0}
        self.parameter_names = ['rate', 'start', 'duration']

    start = _new_property('rates', 'start', ms)
    rate = _new_property('rates', 'rate', Hz)
    duration = _new_property('rates', 'duration', ms)
    
            
class MultipleSpikeGeneratorGroupWithDelays(brian.MultipleSpikeGeneratorGroup):
   
    def __init__(self, spiketimes):
        thresh = brian.directcontrol.MultipleSpikeGeneratorThreshold(spiketimes)
        brian.NeuronGroup.__init__(self, len(spiketimes),
                                   model=brian.LazyStateUpdater(),
                                   threshold=thresh,
                                   clock=state.simclock,
                                   max_delay=state.max_delay*ms)
        self.parameter_names = ['spiketimes']

    def _get_spiketimes(self):
        return self._threshold.spiketimes
    def _set_spiketimes(self, spiketimes):
        assert common.is_listlike(spiketimes)
        if len(spiketimes) == 0 or common.is_number(spiketimes[0]):
            spiketimes = [spiketimes for i in xrange(len(self))]
        assert len(spiketimes) == len(self), "spiketimes (length %d) must contain as many iterables as there are cells in the group (%d)." % (len(spiketimes), len(self))
        self._threshold.set_spike_times(spiketimes)
    spiketimes = property(fget=_get_spiketimes, fset=_set_spiketimes)


class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object.
    """

    non_parameter_attributes = list(common.IDMixin.non_parameter_attributes) + ['parent_group']

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        params = {}
        assert hasattr(self.parent_group, "parameter_names"), str(self.cellclass)
        for name in self.parent_group.parameter_names:
            if name in ['v_thresh', 'v_reset', 'tau_refrac', 'start', 'rate', 'duration']:
                # parameter shared among all cells
                params[name] = float(getattr(self.parent_group, name))
            elif name == 'spiketimes':
                params[name] = getattr(self.parent_group, name)[int(self)]
            elif name == 'v_init':
                params[name] = getattr(self.parent_group[int(self)], name)
            else:
                # parameter may vary from cell to cell
                try:
                    params[name] = float(getattr(self.parent_group[int(self)], name)[0])
                except TypeError, errmsg:
                    raise TypeError("%s. celltype=%s, parameter name=%s" % (errmsg, self.cellclass, name))
        return params
    
    def set_native_parameters(self, parameters):
        for name, value in parameters.items():
            if name in ['v_thresh', 'v_reset', 'tau_refrac', 'start', 'rate', 'duration']:
                setattr(self.parent_group, name, value)
                logging.warning("This parameter cannot be set for individual cells within a Population. Changing the value for all cells in the Population.")
            elif name == 'spiketimes':
                #setattr(self.parent_group, name, [value]*len(self.parent_group))
                self.parent_group.spiketimes[int(self)] = value
                #except IndexError, errmsg:
                #    raise IndexError("%s. index=%d, self.parent_group.spiketimes=%s" % (errmsg, int(self), self.parent_group.spiketimes))
            else:
                setattr(self.parent_group[int(self)], name, value)

class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
  
    numpy1_1_formats = {'spikes': "%d\t%g",
                        'v': "%d\t%g\t%g"}
    numpy1_0_formats = {'spikes': "%g", # only later versions of numpy support different
                        'v': "%g"}      # formats for different columns
    formats = {'spikes': 'id t',
               'v': 'id t v',
               'conductance':'id ge gi t'} #???
  
    def __init__(self, variable, population=None, file=None):
        """
        `file` should be one of:
            a file-name,
            `None` (write to a temporary file)
            `False` (write to memory).
        """
        self.variable = variable
        self.filename = file or None
        self.population = population # needed for writing header information
        self.recorded = set([])
        self._device = None # defer creation until first call of record()
    
    def _create_device(self, group):
        if self.variable == 'spikes':
            device = brian.SpikeMonitor(group, record=True)
        else:
            device = brian.StateMonitor(group, self.variable, record=True, clock=state.simclock)
        net.add(device)
        return device
    
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        #update StateMonitor.record and StateMonitor.recordindex
        if self._device is None:
            self._device = self._create_device(ids[0].parent_group)
        self.recorded = self.recorded.union(ids)
        if self.variable is not 'spikes':
            self._device.record = list(self.recorded)
            self._device.recordindex = dict((i,j) for i,j in zip(self._device.record,
                                                             range(len(self._device.record))))
    
    def get(self, gather=False, compatible_output=True):
        """Returns the recorded data."""
        if self.population:
            offset = self.population.first_id
        else:
            offset = 0
        if self.variable == 'spikes':
            data = numpy.array([(id, time/ms) for (id, time) in self._device.spikes if id in self.recorded])
        elif self.variable == 'v':
            values = self._device.values/mV
            times = self._device.times/ms
            data = numpy.empty((0,3))
            for id, row in enumerate(values):
                new_data = numpy.array([numpy.ones(row.shape)*(id-offset), times, row]).T
                data = numpy.concatenate((data, new_data))
        return data
        
    def write(self, file=None, gather=False, compatible_output=True):
        data = self.get(gather)
        filename = file or self.filename
        recording.rename_existing(filename)
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
        if compatible_output:
            recording.write_compatible_output(filename, filename, self.variable,
                                              Recorder.formats[self.variable],
                                              self.population, state.dt)
        
        

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.simclock = None
        self.initialized = False
        self.num_processes = 1
        self.mpi_rank = 0
        self.min_delay = numpy.NaN
        self.max_delay = numpy.NaN
        
    
    def _get_dt(self):
        if self.simclock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return self.simclock.dt/ms
    def _set_dt(self, timestep):
        if self.simclock is None or timestep != self._get_dt():
            self.simclock = brian.Clock(dt=timestep*ms)
    dt = property(fget=_get_dt, fset=_set_dt)

    @property
    def t(self):
        return self.simclock.t/ms

def reset():
    state.simclock.reinit()

def run(simtime):
    """Run the simulation for simtime ms."""
    # The run() command of brian accepts seconds
    net.run(simtime*ms)
    
    
def create_cells(cellclass, cellparams=None, n=1, parent=None):
    """
    Function used by both `create()` and `Population.__init__()`
    """
    # currently, we create a single NeuronGroup for create(), but
    # arguably we should use n NeuronGroups each containing a single cell
    # either that or use the subgroup() method in connect(), etc
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, basestring):  # celltype is not a standard cell
        try:
            eqs = brian.Equations(cellclass)
        except Exception, errmsg:
            raise common.InvalidModelError(errmsg)
        v_thresh   = cellparams['v_thresh']
        v_reset    = cellparams['v_reset']
        tau_refrac = cellparams['tau_refrac']
        brian_cells = brian.NeuronGroup(n,
                                        model=eqs,
                                        threshold=v_thresh,
                                        reset=v_reset,
                                        clock=state.simclock,
                                        compile=True,
                                        max_delay=state.max_delay)
        cell_parameters = cellparams or {}
    elif isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
        celltype = cellclass(cellparams)
        cell_parameters = celltype.parameters
        
        if isinstance(celltype, cells.SpikeSourcePoisson):    
            fct = celltype.fct
            brian_cells = PoissonGroupWithDelays(n, rates=fct)
        elif isinstance(celltype, cells.SpikeSourceArray):
            spike_times = cell_parameters['spiketimes']
            brian_cells = MultipleSpikeGeneratorGroupWithDelays([spike_times for i in xrange(n)])
        else:
            brian_cells = ThresholdNeuronGroup(n, cellclass.eqs)
    else:
        raise Exception("Invalid cell type: %s" % type(cellclass))    

    if cell_parameters:
        for key, value in cell_parameters.items():
            setattr(brian_cells, key, value)
    # should we globally track the IDs used, so as to ensure each cell gets a unique integer? (need only track the max ID)
    cell_list = numpy.array([ID(cell) for cell in xrange(len(brian_cells))], ID)
    for cell in cell_list:
        cell.parent_group = brian_cells
   
    mask_local = numpy.ones((n,), bool) # all cells are local
    first_id = cell_list[0]
    last_id = cell_list[-1]
   
    if parent:
        parent.brian_cells = brian_cells

    net.add(brian_cells)
    return cell_list, mask_local, first_id, last_id


class Connection(object):

    def __init__(self, brian_connection, index):
        # the index is the nth non-zero element
        self.bc = brian_connection
        n_rows,n_cols = self.bc.W.shape
        self.addr = self.__get_address(index)
        self.source, self.target = self.addr
        #print "creating connection with index", index, "and address", self.addr

    def __get_address(self, index):
        count = 0
        for i, row in enumerate(self.bc.W.data):
            new_count = count + len(row)
            if index < new_count:
                j = self.bc.W.rows[i][index-count]
                return (i,j)
            count = new_count
        raise IndexError("connection with index %d requested, but Connection only contains %d connections." % (index, self.bc.W.getnnz()))

    def _set_weight(self, w):
        w = w or ZERO_WEIGHT
        self.bc[self.addr] = w*self.bc.weight_units

    def _get_weight(self):
        return float(self.bc[self.addr]/self.bc.weight_units)

    def _set_delay(self, d):
        self.bc.delay[self.addr] = d*ms

    def _get_delay(self):
        return float(self.bc.delay[self.addr]/ms)

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)
    

def nesteddictwalk(d):
    """
    Walk a nested dict structure, returning all values in all dicts.
    """
    for value1 in d.values():
        if isinstance(value1, dict):
            for value2 in nesteddictwalk(value1):  # recurse into subdict
                yield value2
        else:
            yield value1
            

class ConnectionManager(object):
    """docstring needed."""

    def __init__(self, synapse_model=None, parent=None):
        self.parent = parent
        self.connections = {}
        self.n = 0

    def __getitem__(self, i):
        """Returns a Connection object."""
        j = 0
        for bc in nesteddictwalk(self.connections):
            assert isinstance(bc, brian.Connection), str(bc)
            j_new = j + bc.W.getnnz()
            if i < j_new:
                return Connection(bc, i-j)
            else:
                j = j_new
        raise Exception("No such connection. i=%d. connection object lengths=%s" % (i, str([c.W.getnnz() for c in nesteddictwalk(self.connections)])))
    
    def __len__(self):
        return self.n
    
    def __connection_generator(self):
        for bc in nesteddictwalk(self.connections):
            for j in range(bc.W.getnnz()):
                yield Connection(bc, j)
                
    
    def __iter__(self):
        return self.__connection_generator()
    
    def _get_brian_connection(self, source_group, target_group, synapse_obj, weight_units):
        bc = None
        syn_id = id(synapse_obj)
        src_id = id(source_group)
        tgt_id = id(target_group)
        if syn_id in self.connections:
            if src_id in self.connections[syn_id]:
                if tgt_id in self.connections[syn_id][src_id]:
                    bc = self.connections[syn_id][src_id][tgt_id]
            else:
                self.connections[syn_id][src_id] = {}
        else:
            self.connections[syn_id] = {src_id: {}}
        if bc is None:
            assert isinstance(source_group, brian.NeuronGroup)
            assert isinstance(target_group, brian.NeuronGroup), type(target_group)
            assert isinstance(synapse_obj, basestring), "%s (%s)" % (synapse_obj, type(synapse_obj))
            bc = brian.DelayConnection(source_group,
                                       target_group,
                                       synapse_obj,
                                       max_delay=state.max_delay)
            bc.weight_units = weight_units
            net.add(bc)
            self.connections[syn_id][src_id][tgt_id] = bc
        return bc
    
    def connect(self, source, targets, weights, delays, synapse_type):
        """
        Connect a neuron to one or more other neurons.
        """
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not common.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        assert isinstance(source, common.IDMixin), str(type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise common.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        if common.is_conductance(targets[0]):
            units = uS
        else:
            units = nA
            
        synapse_type = synapse_type or "excitatory"
        synapse_obj  = targets[0].cellclass.synapses[synapse_type]
        try:
            source_group = source.parent_group
        except AttributeError, errmsg:
            raise common.ConnectionError("%s. Maybe trying to connect from non-existing cell (ID=%s)." % (errmsg, source))
        target_group = targets[0].parent_group # we assume here all the targets belong to the same NeuronGroup
        bc = self._get_brian_connection(source_group,
                                        target_group,
                                        synapse_obj,
                                        units)
        #W = brian.connection.SparseMatrix(shape=(1,len(target_group)), dtype=float)
        W = scipy.sparse.lil_matrix((len(source_group), len(target_group)), dtype=float)
        D = scipy.sparse.lil_matrix((len(source_group), len(target_group)), dtype=float)
        
        src = int(source)
        for tgt, w, d in zip(targets, weights, delays):
            w = w or ZERO_WEIGHT # since sparse matrices use 0 for empty entries, we use this value to mark a connection as existing but of zero weight
            W[src, int(tgt)] = w*units
            D[src, int(tgt)] = d*ms
        bc.connect(W=W)
        bc.delayvec[0:len(bc.source),0:len(bc.target)] = D
        self.n += len(targets)
        
    def get(self, parameter_name, format, offset=(0,0)):
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        synapse_obj = self.parent.post.celltype.synapses[self.parent.target or "excitatory"]
        weight_units = uS and ("cond" in self.parent.post.celltype.__class__.__name__) or nA
        bc = self._get_brian_connection(self.parent.pre.brian_cells,
                                        self.parent.post.brian_cells,
                                        synapse_obj,
                                        weight_units)
        if parameter_name == "weight":
            M = bc.W
            units = weight_units
        elif parameter_name == 'delay':
            M = bc.delay
            units = ms
        else:
            raise Exception("Getting parameters other than weight and delay not yet supported.")
        
        values = M.todense()        
        values = numpy.where(values==0, numpy.nan, values)
        mask = values>0
        values = numpy.where(values<=ZERO_WEIGHT, 0.0, values)
        values /= units
        if format == 'list':
            values = values[mask].tolist()
        elif format == 'array':
            pass
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values
        
    def set(self, name, value):
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        synapse_obj = self.parent.post.celltype.synapses[self.parent.target or "excitatory"]
        weight_units = uS and ("cond" in self.parent.post.celltype.__class__.__name__) or nA
        bc = self._get_brian_connection(self.parent.pre.brian_cells,
                                        self.parent.post.brian_cells,
                                        synapse_obj,
                                        weight_units)
        if name == 'weight':
            M = bc.W
            units = weight_units
        elif name == 'delay':
            M = bc.delay
            units = ms
        else:
            raise Exception("Setting parameters other than weight and delay not yet supported.")
        if common.is_number(value):
            for row in M.data:
                for i in range(len(row)):
                    row[i] = value*units
        elif common.is_listlike(value):
            assert len(value) == M.getnnz()
            address_gen = ((i,j) for i,row in enumerate(bc.W.rows) for j in row)
            for ((i,j),val) in izip(address_gen, value):
                M[i,j] = val*units
        else:
            raise Exception("Values must be scalars or lists/arrays")
                
state = _State()  # a Singleton, so only a single instance ever exists
del _State
net = brian.Network()