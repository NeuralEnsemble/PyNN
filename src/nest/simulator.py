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
from pyNN import common, random
import logging
import numpy
import os

CHECK_CONNECTIONS = False
recorder_list = []

# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.running     = False
        self.optimize    = False

    @property
    def t(self):
        return nest.GetKernelStatus()['time']
    
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
    if not state.running:
        simtime += state.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
        state.running = True
    nest.Simulate(simtime)
    

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
        nest.SetStatus([self], [parameters])


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
        raise common.InvalidModelError(errmsg)
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
        src = self.parent.sources[self.index]
        port = self.parent.ports[self.index]
        synapse_model = self.parent.synapse_model
        return [[src], synapse_model, port]

    @property
    def source(self):
        """The ID of the pre-synaptic neuron."""
        return self.parent.sources[self.index]
    
    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        return self.parent.targets[self.index]

    @property
    def port(self):
        """The port number of this connection."""
        return self.parent.ports[self.index]

    def _set_weight(self, w):
        args = self.id() + [{'weight': w*1000.0}]
        nest.SetConnection(*args)

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return 0.001*nest.GetConnection(*self.id())['weight']

    def _set_delay(self, d):
        args = self.id() + [{'delay': d}]
        nest.SetConnection(*args)

    def _get_delay(self):
        """Synaptic delay in ms."""
        return nest.GetConnection(*self.id())['delay']

    weight = property(_get_weight, _set_weight)
    delay  = property(_get_delay, _set_delay)
    

class ConnectionManager:
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_model='static_synapse', parent=None):
        """
        Create a new ConnectionManager.
        
        `synapse_model` -- the NEST synapse model to be used for all connections
                           created with this manager.
        `parent` -- the parent `Projection`, if any.
        """
        self.sources = []
        self.targets = []
        self.ports = []
        self.synapse_model = synapse_model
        self.parent = parent
        if parent is not None:
            assert parent.plasticity_name == self.synapse_model

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if i < len(self):
            return Connection(self, i)
        else:
            raise IndexError("%d > %d" % (i, len(self)-1))
    
    def __len__(self):
        """Return the number of connections on the local MPI node."""
        if state.optimize:
            return nest.GetDefaults(self.synapse_model)['num_connections']
        else:
            return len(self.sources)
    
    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        for i in range(len(self)):
            yield self[i]
    
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
                          be "excitatory", "inhibitory", or `None`, which is
                          treated the same as "excitatory".
        """
        # are we sure the targets are all on the current node?
        if common.is_listlike(source):
            assert len(source) == 1
            source = source[0]
        if not common.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0
        if synapse_type not in ('excitatory', 'inhibitory', None):
            raise common.ConnectionError("synapse_type must be 'excitatory', 'inhibitory', or None (equivalent to 'excitatory')")
        weights = weights*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if synapse_type == 'inhibitory' and common.is_conductance(targets[0]):
            weights = -1*weights # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]
        
        if not state.optimize:        
            initial_ports = {}
            for tgt in targets:
                try:
                    initial_ports[tgt] = nest.FindConnections([source], [tgt], self.synapse_model)['ports']
                except nest.NESTError, e:
                    raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, source, targets, weights, delays, self.synapse_model))
            
            try:
                nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)            
            except nest.NESTError, e:
                raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, source, targets, weights, delays, self.synapse_model))
        
            final_ports = {}
            for tgt in targets:
                final_ports[tgt] = nest.FindConnections([source], [tgt], self.synapse_model)['ports']
        
            #print "\n", state.mpi_rank, source, "initial:", initial_ports, "final:", final_ports        
            #all_new_ports = []
            local_targets = final_ports.keys()
            local_targets.sort()
            for tgt in local_targets:
                #new_ports = final_ports[tgt].difference(initial_ports[tgt])
                new_ports = final_ports[tgt][len(initial_ports[tgt]):]
                #if state.mpi_rank == 0:
                #    print "-", state.mpi_rank, tgt, initial_ports[tgt], final_ports[tgt], new_ports
                n = len(new_ports)
                if n > 0:   
                    self.sources.extend([source]*n)
                    self.targets.extend([tgt]*n)     
                    self.ports.extend(new_ports)
                #all_new_ports.extend(new_ports)
                
            #print state.mpi_rank, source, targets, all_new_ports, nest.GetConnections([source], self.synapse_model)[0]['targets']
        elif state.optimize:
            try:
                nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)            
            except nest.NESTError, e:
                raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, source, targets, weights, delays, self.synapse_model))
        
    
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
        # this is a slow implementation, going through each connection one at a time
        # better to use GetConnections, which means we should probably store
        # connections in a dict, with source as keys and a list of ports as values
        if format == 'list':
            values = []
            for src, port in zip(self.sources, self.ports):
                value = nest.GetConnection([src], self.synapse_model, port)[parameter_name]
                if parameter_name == "weight":
                    value *= 0.001
                values.append(value)
        elif format == 'array':
            values = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            for src, tgt, port in zip(self.sources, self.targets, self.ports):
                # could instead get tgt from the 'target' entry with GetConnection
                value = nest.GetConnection([src], self.synapse_model, port)[parameter_name]
                # don't need to pass offset as arg, now we store the parent projection
                # (offset is always 0,0 for connections created with connect())
                addr = (src-offset[0], tgt-offset[1])
                if numpy.isnan(values[addr]):
                    values[addr] = value
                else:
                    values[addr] += value
            if parameter_name == 'weight':
                values *= 0.001
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values
    
    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.
        
        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections.
        """
        if not (common.is_number(value) or common.is_listlike(value)):
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
        
        plural_name = name + 's'
        
        source_cells = list(set(self.sources))
        source_cells.sort()
        source_array = numpy.array(self.sources)  # these values
        target_array = numpy.array(self.targets)  # are all for connections
        port_array   = numpy.array(self.ports)      # on the local node        
        
        if common.is_listlike(value):
            assert len(value) == len(port_array)
            value = numpy.array(value)
        if name == 'weight':
            value *= 1000.0
        elif name == 'delay':
            pass
        else:
            translation = self.parent.synapse_dynamics.reverse_translate({name: value})
            name, value = translation.items()[0]
        
        i = 0
        for src in source_cells:
            connection_dict = nest.GetConnections([src], self.synapse_model)[0]
            #print connection_dict
            # obtain arrays of all targets, and current values, for this source
            all_targets = numpy.array(connection_dict['targets'])
            all_values = numpy.array(connection_dict[plural_name])
            # extract the ports that are relevant to this source
            this_source = source_array==src
            ports = port_array[this_source]
            # determine current values just for the local MPI node
            local_targets = target_array[this_source]
            local_mask = numpy.array([tgt in local_targets for tgt in all_targets])
            local_values = all_values[local_mask]
            if common.is_number(value):
                local_new_values = value
            else:
                local_new_values = value[i:i+len(ports)]
                i += len(ports)
            # now set the new values for the local connections 
            local_values[ports] = local_new_values
            nest.SetConnections([src], self.synapse_model, [{plural_name: local_values.tolist()}])
            
        


# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State