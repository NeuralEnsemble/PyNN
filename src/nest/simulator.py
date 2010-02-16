# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    create_cells()
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

import nest
from pyNN import common, random, errors
import logging
import numpy
import os
import sys

CHECK_CONNECTIONS = False
recorder_list = []

logger = logging.getLogger("PyNN")

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.running     = False
        self.optimize    = False
        self.nominal_time = 0.0

    @property
    def t(self):
        #return nest.GetKernelStatus()['time']
        return self.nominal_time
    
    dt = property(fget=lambda self: nest.GetKernelStatus()['resolution'],
                  fset=lambda self, timestep: nest.SetKernelStatus({'resolution': timestep}))    
    
    @property
    def min_delay(self):
        return nest.GetDefaults('static_synapse')['min_delay']
    
    @property
    def max_delay(self):
        # any reason why not nest.GetKernelStatus()['min_delay']?
        return nest.GetDefaults('static_synapse')['max_delay']
    
    @property
    def num_processes(self):
        return nest.GetKernelStatus()['num_processes']
    
    @property
    def mpi_rank(self):
        return nest.Rank()
    
    @property
    def num_threads(self):
        return nest.GetKernelStatus()['local_num_threads']


def run(simtime):
    """Advance the simulation for a certain time."""
    state.nominal_time += simtime
    if not state.running:
        simtime += state.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
        state.running = True
    nest.Simulate(simtime)
    
def reset():
    nest.ResetNetwork()
    state.running = False
    state.nominal_time = 0.0

# --- For implementation of access to individual neurons' parameters ----------- 

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)
        self._v_init = None

    def get_native_parameters(self):
        """Return a dictionary of parameters for the NEST cell model."""
        parameters = nest.GetStatus([int(self)])[0]
        if self._v_init is not None:
            parameters['v_init'] = self._v_init
        return parameters

    def set_native_parameters(self, parameters):
        """Set parameters of the NEST cell model from a dictionary."""
        if 'v_init' in parameters:
            self._v_init = parameters.pop('v_init')
            parameters['V_m'] = self._v_init # not correct, since could set v_init in the middle of a simulation, but until we add a proper reset mechanism, this will do.
        try:
            nest.SetStatus([self], [parameters])
        except: # I can't seem to catch the NESTError that is raised, hence this roundabout way of doing it.
            exc_type, exc_value, traceback = sys.exc_info()
            if exc_type == 'NESTError' and "Unsupported Numpy array type" in exc_value:
                raise errors.InvalidParameterValueError()
            else:
                raise


# --- For implementation of create() and Population.__init__() -----------------

def create_cells(cellclass, cellparams=None, n=1, parent=None):
    """
    Create cells in NEST.
    
    `cellclass`  -- a PyNN standard cell or the name of a native NEST cell model.
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
    if isinstance(cellclass, basestring):  # celltype is not a standard cell
        nest_model = cellclass
        cell_parameters = cellparams or {}
    elif isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
        celltype = cellclass(cellparams)
        nest_model = celltype.nest_name
        cell_parameters = celltype.parameters
    else:
        raise Exception("Invalid cell type: %s" % type(cellclass))
    try:
        cell_gids = nest.Create(nest_model, n)
    except nest.NESTError, errmsg:
        raise errors.InvalidModelError(errmsg)
    if cell_parameters:
        try:
            v_init = cell_parameters.pop('v_init', None)
            if v_init is not None:
                cell_parameters['V_m'] = v_init
            nest.SetStatus(cell_gids, [cell_parameters])
        except nest.NESTError:
            print "NEST error when trying to set the following dictionary: %s" % cell_parameters
            raise
    first_id = cell_gids[0]
    last_id = cell_gids[-1]
    mask_local = numpy.array(nest.GetStatus(cell_gids, 'local'))
    cell_gids = numpy.array([ID(gid) for gid in cell_gids], ID)
    for gid, local in zip(cell_gids, mask_local):
        gid.local = local
    if cell_parameters and v_init is not None:
        for cell in cell_gids:
            cell._v_init = v_init
    return cell_gids, mask_local, first_id, last_id


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, parent, index):
        """
        Create a new connection interface.
        
        `parent` -- a ConnectionManager instance.
        `index` -- the index of this connection in the parent.
        """
        self.parent = parent
        self.index = index

    def id(self):
        """Return a tuple of arguments for `nest.GetConnection()`.
        """
        return self.parent.connections[self.index]

    @property
    def source(self):
        """The ID of the pre-synaptic neuron."""
        src = ID(nest.GetStatus([self.id()], 'source')[0])
        src.parent = self.parent.parent.pre
        src.local = nest.GetStatus([src], 'local')[0]
        return src
    
    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        tgt = ID(nest.GetStatus([self.id()], 'target')[0])
        tgt.parent = self.parent.parent.pre
        tgt.local = nest.GetStatus([tgt], 'local')[0]
        return tgt

    def _set_weight(self, w):
        nest.SetStatus([self.id()], 'weight', w*1000.0)

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        w_nA = nest.GetStatus([self.id()], 'weight')[0]
        if self.parent.synapse_type == 'inhibitory' and common.is_conductance(self.target):
            w_nA *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
        return 0.001*w_nA

    def _set_delay(self, d):
        nest.SetStatus([self.id()], 'delay', d)

    def _get_delay(self):
        """Synaptic delay in ms."""
        return nest.GetStatus([self.id()], 'delay')[0]

    weight = property(_get_weight, _set_weight)
    delay  = property(_get_delay, _set_delay)
    

class ConnectionManager:
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_type` -- the 'physiological type' of the synapse, e.g.
                          'excitatory' or 'inhibitory'
        `synapse_model` -- the NEST synapse model to be used for all connections
                           created with this manager.
        `parent` -- the parent `Projection`, if any.
        """
        self.sources = []
        if synapse_model is None:
            self.synapse_model = 'static_synapse_%s' % id(self)
            nest.CopyModel('static_synapse', self.synapse_model)
        else:
            self.synapse_model = synapse_model
        self.synapse_type = synapse_type
        self.parent = parent
        if parent is not None:
            assert parent.plasticity_name == self.synapse_model
        self._connections = None

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return Connection(self, i)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [Connection(self, j) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))
            
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return nest.GetDefaults(self.synapse_model)['num_connections']
    
    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        for i in range(len(self)):
            yield self[i]

    @property
    def connections(self):
        if self._connections is None:
            self._connections = nest.FindConnections(self.sources, synapse_type=self.synapse_model)
        return self._connections
    
    def connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons.
        
        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        # are we sure the targets are all on the current node?
        if common.is_listlike(source):
            assert len(source) == 1
            source = source[0]
        if not common.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0
        if self.synapse_type not in ('excitatory', 'inhibitory', None):
            raise errors.ConnectionError("synapse_type must be 'excitatory', 'inhibitory', or None (equivalent to 'excitatory')")
        weights = weights*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and common.is_conductance(targets[0]):
            weights = -1*weights # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]
        
        try:
            nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)            
        except nest.NESTError, e:
            raise errors.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, source, targets, weights, delays, self.synapse_model))
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self.sources.append(source)

    def convergent_connect(self, sources, target, weights, delays):
        """
        Connect one or more neurons to a single post-synaptic neuron.
        `sources` -- a list/1D array of pre-synaptic cell IDs, or a single ID.
        `target`  -- the ID of the post-synaptic cell.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        # are we sure the targets are all on the current node?
        if common.is_listlike(target):
            assert len(target) == 1
            target = target[0]
        if not common.is_listlike(sources):
            sources = [sources]
        assert len(sources) > 0, sources
        if self.synapse_type not in ('excitatory', 'inhibitory', None):
            raise errors.ConnectionError("synapse_type must be 'excitatory', 'inhibitory', or None (equivalent to 'excitatory')")
        weights = weights*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and common.is_conductance(target):
            weights = -1*weights # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]
               
        try:
            nest.ConvergentConnect(sources, [target], weights, delays, self.synapse_model)            
        except nest.NESTError, e:
            raise errors.ConnectionError("%s. sources=%s, target=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, sources, target, weights, delays, self.synapse_model))
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self.sources.extend(sources)
    
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
        
        if parameter_name not in ('weight', 'delay'):
            translated_name = None
            if self.parent.synapse_dynamics.fast and parameter_name in self.parent.synapse_dynamics.fast.translations:
                translated_name = self.parent.synapse_dynamics.fast.translations[parameter_name]["translated_name"] # this is a hack that works because there are no units conversions
            elif self.parent.synapse_dynamics.slow:
                for component_name in "timing_dependence", "weight_dependence", "voltage_dependence":
                    component = getattr(self.parent.synapse_dynamics.slow, component_name)
                    if component and parameter_name in component.translations:
                        translated_name = component.translations[parameter_name]["translated_name"]
                        break
            if translated_name:
                parameter_name = translated_name
            else:
                raise Exception("synapse type does not have an attribute '%s', or else this attribute is not accessible." % parameter_name)
        if format == 'list':
            values = nest.GetStatus(self.connections, parameter_name)
            if parameter_name == "weight":
                values = [0.001*val for val in values]
        elif format == 'array':
            value_arr = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            connection_parameters = nest.GetStatus(self.connections)
            for conn in connection_parameters: 
                # don't need to pass offset as arg, now we store the parent projection
                # (offset is always 0,0 for connections created with connect())
                src = conn['source']
                tgt = conn['target']
                value = conn[parameter_name]
                addr = (src-offset[0], tgt-offset[1])
                if numpy.isnan(value_arr[addr]):
                    value_arr[addr] = value
                else:
                    value_arr[addr] += value
            if parameter_name == 'weight':
                value_arr *= 0.001
                if self.synapse_type == 'inhibitory' and common.is_conductance(self[0].target):
                    value_arr *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
            values = value_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values
    
    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if not (common.is_number(value) or common.is_listlike(value)):
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")   
        
        if isinstance(value, numpy.ndarray) and len(value.shape) == 2:
            offset = (self.parent.pre.first_id, self.parent.post.first_id)
            value_list = []
            connection_parameters = nest.GetStatus(self.connections)
            for conn in connection_parameters: 
                addr = (conn['source']-offset[0], conn['target']-offset[1])
                try:
                    val = value[addr]
                except IndexError, e:
                    raise IndexError("%s. addr=%s" % (e, addr))
                if numpy.isnan(val):
                    raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.target))
                else:
                    value_list.append(val)
            value = value_list
        if common.is_listlike(value):
            value = numpy.array(value)
        else:
            value = float(value)

        if name == 'weight':
            value *= 1000.0
            if self.synapse_type == 'inhibitory' and common.is_conductance(self[0].target):
                value *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        elif name == 'delay':
            pass
        else:
            #translation = self.parent.synapse_dynamics.reverse_translate({name: value})
            #name, value = translation.items()[0]
            translated_name = None
            if self.parent.synapse_dynamics.fast:
                if name in self.parent.synapse_dynamics.fast.translations:
                    translated_name = self.parent.synapse_dynamics.fast.translations[name]["translated_name"] # a hack
            if translated_name is None:
                if self.parent.synapse_dynamics.slow:
                    for component_name in "timing_dependence", "weight_dependence", "voltage_dependence":
                        component = getattr(self.parent.synapse_dynamics.slow, component_name)
                        if component and name in component.translations:
                            translated_name = component.translations[name]["translated_name"]
                            break
            if translated_name:
                name = translated_name
        
        i = 0
        try:
            nest.SetStatus(self.connections, name, value)
        except nest.NESTError, e:
            n = 1
            if hasattr(value, '__len__'):
                n = len(value)
            raise Exception("%s. Trying to set %d values." % (e, n))        

# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State
