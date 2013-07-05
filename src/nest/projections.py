# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import nest
import logging
from itertools import repeat
from pyNN import common, errors, core, recording
from pyNN.random import RandomDistribution
from pyNN.space import Space
from . import simulator
from .standardmodels.synapses import StaticSynapse

logger = logging.getLogger("PyNN")


def listify(obj):
    if isinstance(obj, numpy.ndarray):
        return obj.astype(float).tolist()
    elif numpy.isscalar(obj):
        return float(obj)  # NEST chokes on numpy's float types
    else:
        return obj


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
        self.nest_synapse_model = self.synapse_type._get_nest_synapse_model("projection_%d" % Projection._nProj)
        self.synapse_type._set_tau_minus(self.post.local_cells)
        self._sources = []
        self._connections = None

        # Create connections
        connector.connect(self)

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return simulator.Connection(self, i)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [simulator.Connection(self, j) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return nest.GetDefaults(self.nest_synapse_model)['num_connections']

    @property
    def connections(self):
        if self._connections is None:
            self._sources = numpy.unique(self._sources)
            self._connections = nest.GetConnections(self._sources.tolist(), synapse_model=self.nest_synapse_model)
        return self._connections

    def _set_tsodyks_params(self):
        if 'tsodyks' in self.nest_synapse_model: # there should be a better way to do this. In particular, if the synaptic time constant is changed
                                            # after creating the Projection, tau_psc ought to be changed as well.
            assert self.synapse_type in ('excitatory', 'inhibitory'), "only basic synapse types support Tsodyks-Markram connections"
            logger.debug("setting tau_psc")
            targets = nest.GetStatus(self.connections, 'target')
            if self.synapse_type == 'inhibitory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_I']['translated_name']
            if self.synapse_type == 'excitatory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_E']['translated_name']
            tau_syn = nest.GetStatus(targets, (param_name))
            nest.SetStatus(self.connections, 'tau_psc', tau_syn)

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `sources`  -- a 1D array of pre-synaptic cell IDs
        `target`   -- the ID of the post-synaptic cell.

        TO UPDATE
        """
        #logger.debug("Connecting to index %s from %s with %s" % (postsynaptic_index, presynaptic_indices, connection_parameters))
        presynaptic_cells = self.pre[presynaptic_indices].all_cells
        postsynaptic_cell = self.post[postsynaptic_index]
        assert len(presynaptic_cells) > 0, presynaptic_cells
        weights = connection_parameters.pop('weight')
        if self.receptor_type == 'inhibitory' and self.post.conductance_based:
            weights *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        if hasattr(self.post.celltype, "receptor_scale"):  # this is a bit of a hack
            weights *= self.post.celltype.receptor_scale   # needed for the Izhikevich model
        delays = connection_parameters.pop('delay')
        if postsynaptic_cell.celltype.standard_receptor_type:
            try:
                nest.ConvergentConnect(presynaptic_cells.astype(int).tolist(),
                                       [int(postsynaptic_cell)],
                                       listify(weights),
                                       listify(delays),
                                       self.nest_synapse_model)
            except nest.NESTError, e:
                raise errors.ConnectionError("%s. presynaptic_cells=%s, postsynaptic_cell=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, presynaptic_cells, postsynaptic_cell, weights, delays, self.nest_synapse_model))
        else:
            receptor_type = postsynaptic_cell.celltype.get_receptor_type(self.receptor_type)
            if numpy.isscalar(weights):
                weights = repeat(weights)
            if numpy.isscalar(delays):
                delays = repeat(delays)
            for pre, w, d in zip(presynaptic_cells, weights, delays):
                nest.Connect([pre], [postsynaptic_cell], 
                             {'weight': w, 'delay': d, 'receptor_type': receptor_type},
                             model=self.nest_synapse_model)
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self._sources.extend(presynaptic_cells)
        connection_parameters.pop('tau_minus', None)  # TODO: set tau_minus on the post-synaptic cells
        connection_parameters.pop('dendritic_delay_fraction', None)
        connection_parameters.pop('w_min_always_zero_in_NEST', None)
        if connection_parameters:
            #logger.debug(connection_parameters)
            connections = nest.GetConnections(source=presynaptic_cells.astype(int).tolist(),
                                              target=[int(postsynaptic_cell)],
                                              synapse_model=self.nest_synapse_model)
            for name, value in connection_parameters.items():
                nest.SetStatus(connections, name, value)

    def _set_attributes(self, parameter_space):
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))  # only columns for connections that exist on this machine
        for postsynaptic_cell, connection_parameters in zip(self.post.local_cells,
                                                            parameter_space.columns()):
            connections = nest.GetConnections(source=numpy.unique(self._sources).tolist(),
                                              target=[postsynaptic_cell],
                                              synapse_model=self.nest_synapse_model)
            source_mask = numpy.array([numpy.where(x[0]==self._sources)[0][0] for x in connections])
            for name, value in connection_parameters.items():
                nest.SetStatus(connections, name, value[source_mask])

    #def saveConnections(self, file, gather=True, compatible_output=True):
    #    """
    #    Save connections to file in a format suitable for reading in with a
    #    FromFileConnector.
    #    """
    #    import operator
    #
    #    if isinstance(file, basestring):
    #        file = recording.files.StandardTextFile(file, mode='w')
    #
    #    lines   = nest.GetStatus(self.connections, ('source', 'target', 'weight', 'delay'))
    #    if gather == True and simulator.state.num_processes > 1:
    #        all_lines = { simulator.state.mpi_rank: lines }
    #        all_lines = recording.gather_dict(all_lines)
    #        if simulator.state.mpi_rank == 0:
    #            lines = reduce(operator.add, all_lines.values())
    #    elif simulator.state.num_processes > 1:
    #        file.rename('%s.%d' % (file.name, simulator.state.mpi_rank))
    #    logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)
    #
    #    if gather == False or simulator.state.mpi_rank == 0:
    #        lines       = numpy.array(lines, dtype=float)
    #        lines[:,2] *= 0.001
    #        if compatible_output:
    #            lines[:,0] = self.pre.id_to_index(lines[:,0])
    #            lines[:,1] = self.post.id_to_index(lines[:,1])
    #        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
    #        file.close()

    def _get_attributes_as_list(self, *names):
        nest_names = []
        for name in names:
            if name == 'presynaptic_index':
                nest_names.append('source')
            elif name == 'postsynaptic_index':
                nest_names.append('target')
            else:
                nest_names.append(name)
        values = nest.GetStatus(self.connections, nest_names)
        if 'weight' in names: # other attributes could also have scale factors - need to use translation mechanisms
            values = numpy.array(values) # ought to preserve int type for source, target
            scale_factors = numpy.ones(len(names))
            scale_factors[names.index('weight')] = 0.001
            values *= scale_factors
            values = values.tolist()
        if 'presynaptic_index' in names:
            values = numpy.array(values)
            values[:, names.index('presynaptic_index')] -= self.pre.first_id
            values = values.tolist()
        if 'postsynaptic_index' in names:
            values = numpy.array(values)
            values[:, names.index('postsynaptic_index')] -= self.post.first_id
            values = values.tolist()
        return values        

    def _get_attributes_as_arrays(self, *names):
        all_values = []
        for attribute_name in names:
            if attribute_name[-1] == "s":  # weights --> weight, delays --> delay
                attribute_name = attribute_name[:-1]
            value_arr = numpy.nan * numpy.ones((self.pre.size, self.post.size))
            connection_attributes = nest.GetStatus(self.connections, ('source', 'target', attribute_name))
            for conn in connection_attributes:
                # (offset is always 0,0 for connections created with connect())
                src, tgt, value = conn
                addr = self.pre.id_to_index(src), self.post.id_to_index(tgt)
                if numpy.isnan(value_arr[addr]):
                    value_arr[addr] = value
                else:
                    value_arr[addr] += value
            if attribute_name == 'weight':
                value_arr *= 0.001
                if self.synapse_type == 'inhibitory' and self.post.conductance_based:
                    value_arr *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
            all_values.append(value_arr)
        return all_values
