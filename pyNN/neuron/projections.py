# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
from copy import deepcopy
import logging
from itertools import repeat
from collections import defaultdict

import numpy as np

from .. import common, errors, core
from ..space import Space
from . import simulator
from .standardmodels.synapses import StaticSynapse
from ..morphology import MorphologyFilter


logger = logging.getLogger("PyNN")

# if a Projection is created but not assigned to a variable,
# the connections will not exist, so we store a reference here
_projections = []


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator
    _static_synapse_class = StaticSynapse

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type=None, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)
        self._connections = dict((index, defaultdict(list))
                                 for index in self.post._mask_local.nonzero()[0])
        connector.connect(self)
        self._presynaptic_components = dict((index, {}) for index in
                                            self.pre._mask_local.nonzero()[0])
        if self.synapse_type.presynaptic_type:
            self._configure_presynaptic_components()
        _projections.append(self)
        logger.info("--- Projection[%s].__init__() ---" % self.label)

    @property
    def connections(self):
        for x in self._connections.values():
            for y in x.values():
                for z in y:
                    yield z

    def __getitem__(self, i):
        if isinstance(i, int):
            if i < len(self):
                return self.connections[i]
            else:
                raise IndexError("%d > %d" % (i, len(self) - 1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [self.connections[j] for j in range(*i.indices(i.stop))]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self) - 1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(list(self.connections))

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            location_selector=None,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `presynaptic_cells`     -- a 1D array of pre-synaptic cell IDs
        `postsynaptic_cell`     -- the ID of the post-synaptic cell.
        `connection_parameters` -- each parameter should be either a
                                   1D array of the same length as `sources`, or
                                   a single value.
        """
        postsynaptic_cell = self.post[postsynaptic_index]
        if (
            not isinstance(postsynaptic_cell, int)
            or not (0 <= postsynaptic_cell <= simulator.state.gid_counter)
        ):
            err_msg = "Invalid post-synaptic cell: %s (gid_counter=%d)" % (
                postsynaptic_cell, simulator.state.gid_counter)
            raise errors.ConnectionError(err_msg)
        for name, value in connection_parameters.items():
            if isinstance(value, (float, int)):
                connection_parameters[name] = repeat(value)
        assert postsynaptic_cell.local

        cell_obj = self.post[postsynaptic_index]._cell
        if isinstance(location_selector, MorphologyFilter):
            section_index = location_selector(
                cell_obj.morphology,
                filter_by_section=cell_obj.synaptic_receptors[self.receptor_type].keys()
            )
            target_objects = []
            for sid in section_index:
                target = cell_obj.synaptic_receptors[self.receptor_type].get(sid, None)
                if target:
                    target_objects.append(target[0])
                    # what if there are multiple synapses in a single section? here we just take the first
        elif isinstance(location_selector, str):
            if location_selector in cell_obj.section_labels:
                section_index = cell_obj.section_labels[location_selector]
            elif location_selector == "soma":
                section_index = cell_obj.sections[cell_obj.morphology.soma_index]
            elif location_selector == "all":
                section_index = sum((list(item) for item in cell_obj.section_labels.values()), [])
            else:
                raise ValueError("Cell has no location labelled '{}'".format(location_selector))
            target_objects = []
            for sid in section_index:
                target = cell_obj.synaptic_receptors[self.receptor_type].get(sid, None)
                if target:
                    target_objects.append(target[0])
        elif location_selector is None:  # point neuron model
            if "." in self.receptor_type:
                section, target = self.receptor_type.split(".")
                target_objects = [getattr(getattr(cell_obj, section), target)]
            else:
                target_objects = [getattr(cell_obj, self.receptor_type)]
        else:
            raise ValueError("location selector not supported")

        for pre_idx, values in core.ezip(presynaptic_indices, *connection_parameters.values()):
            parameters = dict(zip(connection_parameters.keys(), values))
            for target_object in target_objects:
                self._connections[postsynaptic_index][pre_idx].append(
                    self.synapse_type.connection_type(self, pre_idx, postsynaptic_index,
                                                      cell_obj, target_object,
                                                      **parameters))

    def _configure_presynaptic_components(self):
        """
        For gap junctions potentially other complex synapse types the presynaptic side of the
        connection also needs to be initiated. This is a little tricky with sources distributed on
        different nodes as the parameters need to be gathered to the node where the source is
        hosted before it can be set
        """
        # Get the list of all connections on all nodes
        conn_list = np.array(self.get(self.synapse_type.get_parameter_names(), 'list',
                                      gather='all', with_address=True))
        # Loop through each of the connections where the presynaptic index (first column) is on
        # the local node
        mask_local = np.array(np.in1d(np.squeeze(conn_list[:, 0]),
                                      np.nonzero(self.pre._mask_local)[0]), dtype=bool)
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
        # Evaluate the parameters for the post-synaptic components
        # (typically the "Connection" object)
        # only columns for connections that exist on this machine
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))
        for connection_group, connection_parameters in zip(self._connections.values(),
                                                           parameter_space.columns()):
            for name, value in connection_parameters.items():
                for index in connection_group:
                    for connection in connection_group[index]:
                        setattr(connection, name, value[index])

    def _set_initial_value_array(self, variable, value):
        raise NotImplementedError
