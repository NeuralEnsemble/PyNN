# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import logging
from itertools import izip, repeat
from pyNN import common, errors, core
from pyNN.random import RandomDistribution, NativeRNG
from . import simulator

logger = logging.getLogger("PyNN")

class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 label=None, rng=None):
        __doc__ = common.Projection.__init__.__doc__
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   label, rng)

        ### Deal with short-term synaptic plasticity
        #if self.synapse_dynamics and self.synapse_dynamics.fast:
        #    # need to check it is actually the Ts-M model, even though that is the only one at present!
        #    parameter_space = self.synapse_dynamics.fast.translated_parameters
        #    parameter_space.evaluate(mask=self.post._mask_local)
        #    for cell, P in zip(self.post, parameter_space):
        #        cell._cell.set_Tsodyks_Markram_synapses(self.synapse_type,
        #                                                P['U'], P['tau_rec'],
        #                                                P['tau_facil'], P['u0'])
        #    self.synapse_model = 'Tsodyks-Markram'
        #else:
        #    self.synapse_model = None
        self.connections = []

        ## Create connections
        connector.connect(self)

        logger.info("--- Projection[%s].__init__() ---" %self.label)

        ### Deal with long-term synaptic plasticity
        #if self.synapse_dynamics and self.synapse_dynamics.slow:
        #    ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
        #    if ddf > 0.5 and self._simulator.state.num_processes > 1:
        #        # depending on delays, can run into problems with the delay from the
        #        # pre-synaptic neuron to the weight-adjuster mechanism being zero.
        #        # The best (only?) solution would be to create connections on the
        #        # node with the pre-synaptic neurons for ddf>0.5 and on the node
        #        # with the post-synaptic neuron (as is done now) for ddf<0.5
        #        raise NotImplementedError("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")

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
        return len(self.connections)

#    def _resolve_synapse_type(self):
#        if self.synapse_type is None:
#            self.synapse_type = weight>=0 and 'excitatory' or 'inhibitory'
#        if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
#            self.synapse_type += '_TM'

    def _convergent_connect(self, presynaptic_cells, postsynaptic_cell,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `presynaptic_cells`     -- a 1D array of pre-synaptic cell IDs
        `postsynaptic_cell`     -- the ID of the post-synaptic cell.
        `connection_parameters` -- each parameter should be either a
                                   1D array of the same length as `sources`, or
                                   a single value.
        """
        logger.debug("Convergent connect. Weights=%s" % connection_parameters['weight'])
        if not isinstance(postsynaptic_cell, int) or postsynaptic_cell > simulator.state.gid_counter or postsynaptic_cell < 0:
            errmsg = "Invalid post-synaptic cell: %s (gid_counter=%d)" % (postsynaptic_cell, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        assert postsynaptic_cell.local
        plasticity_mechanism = self.synapse_type.model

        for pre, values in core.ezip(presynaptic_cells, *connection_parameters.values()):
            parameters = dict(zip(connection_parameters.keys(), values))
            #logging.debug("Connecting neuron #%s to neuron #%s with synapse type %s, weight %g, delay %g", source, receptor_type, self.synapse_type, weight, delay)
            if not isinstance(pre, common.IDMixin):
                raise errors.ConnectionError("Invalid pre-synaptic cell ID: %s" % pre)
            self.connections.append(
                simulator.Connection(pre, postsynaptic_cell, self.receptor_type,
                                     plasticity_mechanism, **parameters))

    # --- Methods for setting connection parameters ----------------------------

    def _set_attributes(self, parameter_space):
        parameter_space.evaluate()  # should really take only columns for connections that exist on this machine
        for connection, connection_parameters in zip(self.connections, parameter_space):
            for name, value in connection_parameters.items():
                setattr(connection, name, value)
    #    for name, value in attributes.items():
    #        if numpy.isscalar(value):
    #            for c in self:
    #                setattr(c, name, value)
    #        elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
    #            for c in self.connections:
    #                addr = (self.pre.id_to_index(c.source), self.post.id_to_index(c.receptor_type))
    #                try:
    #                    val = value[addr]
    #                except IndexError, e:
    #                    raise IndexError("%s. addr=%s" % (e, addr))
    #                if numpy.isnan(val):
    #                    raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.receptor_type))
    #                else:
    #                    setattr(c, name, val)
    #        elif core.is_listlike(value):
    #            for c,val in zip(self.connections, value):
    #                setattr(c, name, val)
    #        elif isinstance(value, RandomDistribution):
    #            if isinstance(value.rng, NativeRNG):
    #                rarr = simulator.nativeRNG_pick(len(self),
    #                                                value.rng,
    #                                                value.name,
    #                                                value.parameters)
    #            else:
    #                rarr = value.next(len(self))
    #            for c,val in zip(self.connections, rarr):
    #                setattr(c, name, val)
    #        else:
    #            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
