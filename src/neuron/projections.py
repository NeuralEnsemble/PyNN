# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
from copy import deepcopy
import numpy
import logging
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
from itertools import repeat, chain
from collections import defaultdict
from pyNN import common, errors, core
from pyNN.random import RandomDistribution, NativeRNG
from pyNN.space import Space
from . import simulator
from .standardmodels.synapses import StaticSynapse, TsodyksMarkramSynapse

logger = logging.getLogger("PyNN")

_projections = []  # if a Projection is created but not assigned to a variable,
                   # the connections will not exist, so we store a reference here

class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator
    _static_synapse_class = StaticSynapse

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type=None, source=None, receptor_type=None,
                 space=Space(), label=None):
        __doc__ = common.Projection.__init__.__doc__
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)
        self._connections = dict((index, defaultdict(list)) for index in self.post._mask_local.nonzero()[0])
        connector.connect(self)
        self._presynaptic_components = dict((index, {}) for index in 
                                            self.pre._mask_local.nonzero()[0])
        if self.synapse_type.presynaptic_type:
            self._configure_presynaptic_components()
        _projections.append(self)
        logger.info("--- Projection[%s].__init__() ---" %self.label)

    @property
    def connections(self):
        for x in self._connections.values():
            for y in x.values():
                for z in y:
                    yield z

    def __getitem__(self, i):
        __doc__ = common.Projection.__getitem__.__doc__
        if isinstance(i, int):
            if i < len(self):
                return self.connections[i]
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [self.connections[j] for j in range(*i.indices(i.stop))]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(list(self.connections))

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `presynaptic_cells`     -- a 1D array of pre-synaptic cell IDs
        `postsynaptic_cell`     -- the ID of the post-synaptic cell.
        `connection_parameters` -- each parameter should be either a
                                   1D array of the same length as `sources`, or
                                   a single value.
        """
        #logger.debug("Convergent connect. Weights=%s" % connection_parameters['weight'])
        postsynaptic_cell = self.post[postsynaptic_index]
        if not isinstance(postsynaptic_cell, int) or postsynaptic_cell > simulator.state.gid_counter or postsynaptic_cell < 0:
            errmsg = "Invalid post-synaptic cell: %s (gid_counter=%d)" % (postsynaptic_cell, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        for name, value in connection_parameters.items():
            if isinstance(value, (float, int)):
                connection_parameters[name] = repeat(value)
        assert postsynaptic_cell.local
        for pre_idx, values in core.ezip(presynaptic_indices, *connection_parameters.values()):
            parameters = dict(zip(connection_parameters.keys(), values))
            #logger.debug("Connecting neuron #%s to neuron #%s with synapse type %s, receptor type %s, parameters %s", pre_idx, postsynaptic_index, self.synapse_type, self.receptor_type, parameters)
            self._connections[postsynaptic_index][pre_idx].append(
                self.synapse_type.connection_type(self, pre_idx, postsynaptic_index, **parameters))

    def _configure_presynaptic_components(self):
        """
        For gap junctions potentially other complex synapse types the presynaptic side of the 
        connection also needs to be initiated. This is a little tricky with sources distributed on
        different nodes as the parameters need to be gathered to the node where the source is 
        hosted before it can be set
        """
        # Get the list of all connections on all nodes
        conn_list = numpy.array(self.get(self.synapse_type.get_parameter_names(), 'list', 
                                               gather='all', with_address=True))
        # Loop through each of the connections where the presynaptic index (first column) is on 
        # the local node
        mask_local = numpy.array(numpy.in1d(numpy.squeeze(conn_list[:,0]), 
                                            numpy.nonzero(self.pre._mask_local)[0]), dtype=bool)
        for conn in conn_list[mask_local, :]:
            pre_idx = int(conn[0])
            post_idx = int(conn[1])
            params = dict(zip(self.synapse_type.get_parameter_names(), conn[2:]))
            self._presynaptic_components[pre_idx][post_idx] = \
                                self.synapse_type.presynaptic_type(self, pre_idx, post_idx, **params)

    def _set_attributes(self, parameter_space):
        # If synapse has pre-synaptic components evaluate the parameters for them
        if self.synapse_type.presynaptic_type:
            presyn_param_space = deepcopy(parameter_space)
            presyn_param_space.evaluate(mask=(slice(None), self.pre._mask_local))
            for component, connection_parameters in zip(self._presynaptic_components.values(), 
                                                        presyn_param_space.columns()):
                for name, value in connection_parameters.items():
                    for index in component:
                        setattr(component[index], name, value[index])
        # Evaluate the parameters for the post-synaptic components (typically the "Connection" object)
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))  # only columns for connections that exist on this machine
        for connection_group, connection_parameters in zip(self._connections.values(),
                                                           parameter_space.columns()):
            for name, value in connection_parameters.items():
                for index in connection_group:
                    for connection in connection_group[index]:
                        setattr(connection, name, value[index])
