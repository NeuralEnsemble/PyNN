# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import logging
from itertools import izip, repeat, chain
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
        self._connections = dict((index, {}) for index in self.post._mask_local.nonzero()[0])           
        connector.connect(self)
        self._configure_sources()
        _projections.append(self)
        logger.info("--- Projection[%s].__init__() ---" %self.label)

    def _configure_sources(self):
        """
        For gap junctions potentially other complex synapse types the presynaptic side of the 
        connection also needs to be initiated. This is a litle tricky with sources distributed on
        different nodes as the parameters need to be gathered to the node where the source is 
        hosted before it can be set
        """
        self._presynaptic_connections = []
        if self.synapse_type.active_source:
            param_values = self.get(parameter.keys(), 'array', gather=True)
            for src in self.pre._mask_local:
                pass
#                urce(src, **dict(zip(parameter.keys(), 
#                                                       [ v[src,:] for v in param_values])))
    @property
    def connections(self):
        for x in self._connections.values():
            for y in x.values():
                yield y

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
            self._connections[postsynaptic_index][pre_idx] = \
                simulator.Connection(self, pre_idx, postsynaptic_index, **parameters)

    def _set_attributes(self, parameter_space):
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))  # only columns for connections that exist on this machine
        for connection_group, connection_parameters in zip(self._connections.values(),
                                                           parameter_space.columns()):
            for name, value in connection_parameters.items():
                for index in connection_group:
                    setattr(connection_group[index], name, value[index])
