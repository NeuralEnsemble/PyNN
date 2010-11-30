# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    create_cells()
    reset()
    run()

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

import logging
import brian
import numpy
from itertools import izip
import scipy.sparse
from pyNN import common, cells, errors, standardmodels, core

mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz

# Global variables
recorder_list = []
ZERO_WEIGHT = 1e-99

logger = logging.getLogger("PyNN")

# --- Internal Brian functionality -------------------------------------------- 

def _new_property(obj_hierarchy, attr_name, units):
    """
    Return a new property, mapping attr_name to obj_hierarchy.attr_name.
    
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

class ThresholdNeuronGroup(brian.NeuronGroup):
    
    def __init__(self, n, equations, threshold=-50*mV, reset=-60*mV, refractory=100*ms):
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   threshold=threshold,
                                   reset=reset,
                                   refractory=refractory, # this is set to a very large value as it acts as a maximum refractoriness
                                   compile=True,
                                   clock=state.simclock,
                                   max_delay=state.max_delay*ms,
                                   )
        self.parameter_names = equations._namespace.keys() + ['v_thresh', 'v_reset', 'tau_refrac']
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        self.initial_values = {}

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)

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
            self._variable_rate = False
            self._S[0,:] = rates
            self._S0[0] = rates
        #self.var_index = {'rate':0}
        self.parameter_names = ['rate', 'start', 'duration']

    start = _new_property('rates', 'start', ms)
    rate = _new_property('rates', 'rate', Hz)
    duration = _new_property('rates', 'duration', ms)
    
    def initialize(self):
        pass
    
            
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
        assert core.is_listlike(spiketimes)
        if len(spiketimes) == 0 or numpy.isscalar(spiketimes[0]):
            spiketimes = [spiketimes for i in xrange(len(self))]
        assert len(spiketimes) == len(self), "spiketimes (length %d) must contain as many iterables as there are cells in the group (%d)." % (len(spiketimes), len(self))
        self._threshold.set_spike_times(spiketimes)
    spiketimes = property(fget=_get_spiketimes, fset=_set_spiketimes)

    def initialize(self):
        pass


# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        """Initialize the simulator."""
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
    """Reset the state of the current network to time t = 0."""
    state.simclock.reinit()
    for device in net.operations:
        if hasattr(device, "reinit"):
            device.reinit()
    for group in net.groups:
        group.reinit()
        group.initialize()

def run(simtime):
    """Advance the simulation for a certain time."""
    # The run() command of brian accepts seconds
    net.run(simtime*ms)
    
    
# --- For implementation of access to individual neurons' parameters -----------
    
class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
    
    def get_native_parameters(self):
        """Return a dictionary of parameters for the Brian cell model."""
        params = {}
        assert hasattr(self.parent_group, "parameter_names"), str(self.cellclass)
        for name in self.parent_group.parameter_names:
            if name in ['v_thresh', 'v_reset', 'tau_refrac', 'start', 'rate', 'duration']:
                # parameter shared among all cells
                params[name] = float(getattr(self.parent_group, name))
            elif name == 'spiketimes':
                params[name] = getattr(self.parent_group, name)[int(self)]
            else:
                # parameter may vary from cell to cell
                try:
                    params[name] = float(getattr(self.parent_group, name)[int(self)])
                except TypeError, errmsg:
                    raise TypeError("%s. celltype=%s, parameter name=%s" % (errmsg, self.cellclass, name))
        return params
    
    def set_native_parameters(self, parameters):
        """Set parameters of the Brian cell model from a dictionary."""
        for name, value in parameters.items():
            if name in ['v_thresh', 'v_reset', 'tau_refrac', 'start', 'rate', 'duration']:
                setattr(self.parent_group, name, value)
                logger.warning("This parameter cannot be set for individual cells within a Population. Changing the value for all cells in the Population.")
            elif name == 'spiketimes':
                all_spiketimes = [st[st>state.t] for st in self.parent_group.spiketimes]
                all_spiketimes[int(self)] = value
                self.parent_group.spiketimes = all_spiketimes
            else:
                setattr(self.parent_group[int(self)], name, value)
        
    def set_initial_value(self, variable, value):
        self.parent_group.initial_values[variable][int(self)] = value
    
    def get_initial_value(self, variable):
        return self.parent_group.initial_values[variable][int(self)]
    

# --- For implementation of create() and Population.__init__() ----------------- 
    
class STDP(brian.STDP):
    '''
    See documentation for brian:class:`STDP` for more details. Options hidden here could be used in a more
    general manner. For example, spike interactions (all-to-all, nearest), or the axonal vs. dendritic delays
    because Brian do support those features....
    '''
    def __init__(self, C, taup, taum, Ap, Am, mu_p, mu_m, wmin=0, wmax=None,
                 delay_pre=None, delay_post=None):
        if wmax is None:
            raise AttributeError, "You must specify the maximum synaptic weight"
        wmax  = float(wmax) # removes units
        wmin  = float(wmin)
        Ap   *= wmax   # removes units
        Am   *= wmax   # removes units
        print Ap, Am, mu_p, mu_m
        eqs = brian.Equations('''
            dA_pre/dt  = -A_pre/taup  : 1
            dA_post/dt = -A_post/taum : 1''', taup=taup, taum=taum, wmax=wmax, mu_m=mu_m, mu_p=mu_p)
        pre   = 'A_pre += Ap'
        pre  += '\nw += A_post*pow(w/wmax, mu_m)'
        
        post  = 'A_post += Am'        
        post += '\nw += A_pre*pow(1-w/wmax, mu_p)'
        brian.STDP.__init__(self, C, eqs=eqs, pre=pre, post=post, wmin=wmin, wmax=wmax, delay_pre=None, delay_post=None, clock=None)

# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """
    
    def __init__(self, brian_connection, index):
        """
        Create a new connection.
        
        `brian_connection` -- a Brian Connection object (may contain
                              many connections).
        `index` -- the index of the current connection within
                   `brian_connection`, i.e. the nth non-zero element
                   in the weight array.
        """
        # the index is the nth non-zero element
        self.bc                  = brian_connection
        self.size                = self.bc.W.getnnz()
        self.addr                = self.__get_address(index)
        self.source, self.target = self.addr
        #print "creating connection with index", index, "and address", self.addr

    def __get_address(self, index):
        """
        Return the (i,j) indices of the element in the weight/delay matrices
        that corresponds to the connection with the given index, i.e. the
        nth non-zero element in the weight array where n=index.
        """
        if index < self.size:
            x, y = self.bc.W.nonzero()
            return (x[index], y[index])
        else:
            raise IndexError("connection with index %d requested, but Connection only contains %d connections." % (index, self.size))

    def _set_weight(self, w):
        w = w or ZERO_WEIGHT
        self.bc[self.addr] = w*self.bc.weight_units

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        ###print "in Connection._get_weight(), weight_units = %s" % self.bc.weight_units
        return float(self.bc[self.addr]/self.bc.weight_units)

    def _set_delay(self, d):
        self.bc.delay[self.addr] = d*ms

    def _get_delay(self):
        """Synaptic delay in ms."""
        if isinstance(self.bc, brian.DelayConnection):
            return float(self.bc.delay[self.addr]/ms)
        if isinstance(self.bc, brian.Connection):
            return float(self.bc.delay * self.bc.source.clock.dt/ms)
            
    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)
    

class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_type` -- the 'physiological type' of the synapse, e.g.
                          'excitatory' or 'inhibitory',or any other key in the
                          `synapses` attibute of the celltype class.
        `synapse_model` -- not used. Present for consistency with other simulators.
        `parent` -- the parent `Projection`, if any.
        """
        self.synapse_type      = synapse_type
        self.synapse_model     = synapse_model
        self.parent            = parent
        self.n                 = 0
        self.brian_connections = None

    def __getitem__(self, i):
        """Return the `i`th connection as a Connection object."""
        j = 0
        assert isinstance(self.brian_connections, brian.Connection), str(self.brian_connections)
        j_new = j + self.brian_connections.W.getnnz()
        if i < j_new:
            return Connection(self.brian_connections, i-j)
        else:
            j = j_new
        raise Exception("No such connection. i=%d. connection object lengths=%s" % (i, str(self.brian_connections.W.getnnz())))
    
    def __len__(self):
        """Return the total number of connections in this manager."""
        return self.n
    
    def __connection_generator(self):
        """Yield each connection in turn."""
        for i in range(self.brian_connections.W.getnnz()):
            yield Connection(self.brian_connections, i)
                
    def __iter__(self):
        """Return an iterator over all connections in this manager."""
        return self.__connection_generator()
    
    def _get_brian_connection(self, source_group, target_group, synapse_obj, weight_units, homogeneous=False):
        """
        Return the Brian Connection object that connects two NeuronGroups with a
        given synapse model.
        
        source_group -- presynaptic Brian NeuronGroup.
        target_group -- postsynaptic Brian NeuronGroup
        synapse_obj  -- name of the variable that will be modified by synaptic
                        input.
        weight_units -- Brian Units object: nA for current-based synapses,
                        uS for conductance-based synapses.
        """
        if self.brian_connections is None:
            assert isinstance(source_group, brian.NeuronGroup)
            assert isinstance(target_group, brian.NeuronGroup), type(target_group)
            assert isinstance(synapse_obj, basestring), "%s (%s)" % (synapse_obj, type(synapse_obj))
            if not homogeneous:
                self.brian_connections = brian.DelayConnection(source_group,
                                                               target_group,
                                                               synapse_obj,
                                                               max_delay=state.max_delay*ms)
            else:
                self.brian_connections = brian.Connection(source_group,
                                                          target_group,
                                                          synapse_obj,
                                                          max_delay=state.max_delay*ms)
            self.brian_connections.weight_units = weight_units
            net.add(self.brian_connections)
        return self.brian_connections
    
    def connect(self, source, targets, weights, delays, homogeneous=False):
        """
        Connect a neuron to one or more other neurons with a static connection.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        if not isinstance(source, common.IDMixin):
            raise errors.ConnectionError("source should be an ID object, actually %s" % type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        if common.is_conductance(targets[0]):
            units = uS
        else:
            units = nA
        synapse_type = self.synapse_type or "excitatory"
        synapse_obj  = targets[0].cellclass.synapses[synapse_type]
        try:
            source_group = source.parent_group
        except AttributeError, errmsg:
            raise errors.ConnectionError("%s. Maybe trying to connect from non-existing cell (ID=%s)." % (errmsg, source))
        target_group = targets[0].parent_group # we assume here all the targets belong to the same NeuronGroup
        bc      = self._get_brian_connection(source_group, target_group, synapse_obj, units, homogeneous)        
        src     = int(source)
        targets = numpy.array(targets, int)
        weights = numpy.array(weights) * units
        delays  = numpy.array(delays) * ms
        weights[weights == 0] = ZERO_WEIGHT
        bc.W.rows[src] = targets
        bc.W.data[src] = weights        
        if not homogeneous:
            bc.delayvec.rows[src] = targets
            bc.delayvec.data[src] = delays
        else:
            bc.delay = int(delays[0] / bc.source.clock.dt)
        self.n += len(targets)
        
    def get(self, parameter_name, format, offset=(0,0)):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this manager.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        `offset` -- not used. Present for consistency with other simulators.
        
        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if such a connection exists. If there are no such connections, X_ij will
        be NaN.
        """
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        bc = self.brian_connections
        if parameter_name == "weight":
            if not hasattr(bc.W, 'alldata'): 
                weights = bc.W.connection_matrix().alldata
            else:
                weights = bc.W.alldata
            values = weights / bc.weight_units            
        elif parameter_name == 'delay':
            if isinstance(bc, brian.DelayConnection):
                if not hasattr(bc.delay, 'alldata'): 
                    delays = bc.delay.connection_matrix().alldata
                else:
                    delays = bc.delay.alldata
                values = delays / ms
            else:
                values = bc.delay * bc.source.clock.dt * numpy.ones(self.brian_connections.W.getnnz()) /ms
        else:
            raise Exception("Getting parameters other than weight and delay not yet supported.")
        
        if format == 'list':
            values = values.tolist()
        elif format == 'array':
            value_arr = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            sources, targets = bc.W.nonzero()
            values_arr[sources, targets] = values
            values = values_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values
        
    def set(self, name, value):
        """
        Set connection attributes for all connections in this manager.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        bc = self.brian_connections
        if name == 'weight':
            M = bc.W
            units = bc.weight_units
        elif name == 'delay':
            M = bc.delay
            units = ms
        else:
            raise Exception("Setting parameters other than weight and delay not yet supported.")
        value = value*units
        if numpy.isscalar(value):
            if (name == 'weight') or (name == 'delay' and isinstance(bc, brian.DelayConnection)):
                for row in M.data:
                    for i in range(len(row)):
                        row[i] = value
            elif (name == 'delay' and isinstance(bc, brian.Connection)):
                bc.delay = int(value / bc.source.clock.dt)
            else:
                raise Exception("Setting a non appropriate parameter")
        elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
            if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                raise Exception("FastConnector have been used, and only fixed homogeneous delays are allowed")
            address_gen = ((i,j) for i,row in enumerate(bc.W.rows) for j in row)
            for (i,j) in address_gen:
                M[i,j] = value[i,j]*units
        elif core.is_listlike(value):
            assert len(value) == M.getnnz()
            if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                raise Exception("FastConnector have been used, and only fixed homogeneous delays are allowed")
            address_gen = ((i,j) for i,row in enumerate(bc.W.rows) for j in row)
            for ((i,j),val) in izip(address_gen, value):
                M[i,j] = val*units
        else:
            raise Exception("Values must be scalars or lists/arrays")
                

# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State
net = brian.Network()
