# -*- coding: utf-8 -*-
"""
NEST v3 implementation of the PyNN API.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
import logging

import numpy as np
import nest

from .. import common, errors
from ..space import Space
from ..parameters import simplify
from . import simulator
from .standardmodels.synapses import StaticSynapse
from .conversion import make_sli_compatible

logger = logging.getLogger("PyNN")


def listify(obj):
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif np.isscalar(obj):
        return float(obj)  # NEST chokes on numpy's float types
    else:
        return obj


def split_array_to_avoid_repeats(arr, **associated_arrays):
    assert arr.dtype == int
    n_sub_arrays = np.bincount(arr).max()
    split_indices = [[] for i in range(n_sub_arrays)]
    index_pointer = defaultdict(int)
    for i, element in enumerate(arr):
        split_index = index_pointer[element]
        split_indices[split_index].append(i)
        index_pointer[element] += 1
    sub_arrays = [arr[split_index] for split_index in split_indices]
    sub_associated = []
    for split_index in split_indices:
        assoc = {}
        for key, value in associated_arrays.items():
            if isinstance(value, np.ndarray):
                assoc[key] = value[split_index]
            else:
                assoc[key] = value
        sub_associated.append(assoc)
    return sub_arrays, sub_associated


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
        self.nest_synapse_model = self.synapse_type._get_nest_synapse_model()
        self.nest_synapse_label = Projection._nProj
        self.synapse_type._set_tau_minus(self.post.local_node_collection)
        self._sources = set()
        self._connections = None
        # This is used to keep track of common synapse properties
        self._common_synapse_properties = {}
        self._common_synapse_property_names = None

        # Create connections
        connector.connect(self)

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return simulator.Connection(self, i)
            else:
                raise IndexError("%d > %d" % (i, len(self) - 1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [simulator.Connection(self, j) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self) - 1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        nest_model = self.post.celltype.nest_name[self._simulator.state.spike_precision]
        local_nodes = nest.GetNodes({"model": nest_model}, local_only=True)
        local_connections = nest.GetConnections(target=local_nodes,
                                                synapse_model=self.nest_synapse_model,
                                                synapse_label=self.nest_synapse_label)
        return len(local_connections)

    @property
    def nest_connections(self):
        if self._connections is None or self._simulator.state.stale_connection_cache:
            if len(self._sources) > 0:
                self._connections = nest.GetConnections(
                    nest.NodeCollection(sorted(self._sources)),
                    synapse_model=self.nest_synapse_model,
                    synapse_label=self.nest_synapse_label)
            else:
                self._connections = []
            self._simulator.state.stale_connection_cache = False
        return self._connections

    @property
    def connections(self):
        """
        Returns an iterator over local connections in this projection, as `Connection` objects.
        """
        return (simulator.Connection(self, i) for i in range(len(self)))

    def _connect(self, rule_params, syn_params):
        """
        Create connections by calling nest. Connect on the presynaptic and postsynaptic population
        with the parameters provided by params.
        """
        if 'tsodyks' in self.nest_synapse_model:
            translations = self.post.local_cells[0].celltype.translations
            if self.receptor_type == 'inhibitory':
                param_name = translations['tau_syn_I']['translated_name']
            elif self.receptor_type == 'excitatory':
                param_name = translations['tau_syn_E']['translated_name']
            else:
                raise NotImplementedError()
            syn_params.update(
                {'tau_psc': nest.GetStatus([self.nest_connections[0, 1]], param_name)})

        syn_params.update({'synapse_label': self.nest_synapse_label})
        nest.Connect(self.pre.node_collection,
                     self.post.node_collection,
                     rule_params, syn_params)
        self._simulator.state.stale_connection_cache = True
        self._sources.update(
            nest.GetConnections(synapse_model=self.nest_synapse_model,
                                synapse_label=self.nest_synapse_label).sources()
        )

    def _identify_common_synapse_properties(self):
        """
        Use the connection between the sample indices to distinguish
        between local and common synapse properties.
        """
        sample_connection = nest.GetConnections(
            # take any source from the set
            source=nest.NodeCollection([next(iter(self._sources))]),
            synapse_model=self.nest_synapse_model,
            synapse_label=self.nest_synapse_label)[:1]

        local_parameters = nest.GetStatus(sample_connection)[0].keys()
        all_parameters = nest.GetDefaults(self.nest_synapse_model).keys()
        self._common_synapse_property_names = [name for name in all_parameters
                                               if name not in local_parameters]

    def _update_syn_params(self, syn_dict, connection_parameters):
        """
        Update the paramaters to be passed in the nest.Connect method with the connection
        parameters specific to the synapse type.

        `syn_dict` - the dictionary to be passed to nest.Connect containing "local" parameters
        `connection_parameters` - a dictionary containing all parameters (local and common)
        """
        # Set connection parameters other than weight and delay
        if connection_parameters:
            for name, value in connection_parameters.items():
                if name not in self._common_synapse_property_names:
                    value = make_sli_compatible(value)
                    if isinstance(value, np.ndarray):
                        syn_dict.update({name: np.array([value.tolist()])})
                    else:
                        syn_dict.update({name: value})
        return syn_dict

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `presynaptic_indices` - 1D array of presynaptic indices
        `postsynaptic_index` - integer - the index of the postsynaptic neuron
        `connection_parameters` - dict whose keys are native NEST parameter names.
                                  Values may be scalars or arrays.
        """
        # Clean the connection parameters by removing parameters that are
        # used by PyNN but should not be passed to NEST
        # TODO: set tau_minus on the post-synaptic cells
        connection_parameters.pop('tau_minus', None)
        connection_parameters.pop('dendritic_delay_fraction', None)
        connection_parameters.pop('w_min_always_zero_in_NEST', None)

        syn_dict = {
            'synapse_model': self.nest_synapse_model,
            'synapse_label': self.nest_synapse_label,
        }

        # Weights require some special handling
        if self.receptor_type == 'inhibitory' and self.post.conductance_based:
            # NEST wants negative values for inhibitory weights, even if these are conductances
            connection_parameters['weight'] *= -1
            if "stdp" in self.nest_synapse_model:
                # just some very large negative value to avoid
                # NEST complaining about weight and Wmax having different signs
                # (see https://github.com/NeuralEnsemble/PyNN/issues/636)
                # Will be overwritten below.
                syn_dict["Wmax"] = -1.2345e6
                connection_parameters["Wmax"] *= -1
        # the following two lines are a bit of a hack, needed for the Izhikevich model
        if hasattr(self.post, "celltype") and hasattr(self.post.celltype, "receptor_scale"):
            connection_parameters['weight'] *= self.post.celltype.receptor_scale

        # Prepare connections. NodeCollections can't have repeated values, so for some
        # connector types we need to split the presynaptic cells into groups that
        # don't have such repeats.

        # note that NEST needs sorted indices
        sort_indices = presynaptic_indices.argsort()
        presynaptic_indices = presynaptic_indices[sort_indices]
        for name, value in connection_parameters.items():
            if isinstance(value, np.ndarray):
                connection_parameters[name] = value[sort_indices]

        try:
            presynaptic_cell_groups = [self.pre.node_collection[presynaptic_indices]]
            connection_parameter_groups = [connection_parameters]
        except ValueError as err:
            if "All node IDs in a NodeCollection have to be unique" in str(err):
                presynaptic_index_groups, connection_parameter_groups = \
                    split_array_to_avoid_repeats(presynaptic_indices, **connection_parameters)
                presynaptic_cell_groups = [self.pre.node_collection[i]
                                           for i in presynaptic_index_groups]
            else:
                raise
        postsynaptic_cell = self.post[postsynaptic_index]

        # Create connections and set parameters
        for presynaptic_cells, connection_parameter_group in zip(presynaptic_cell_groups,
                                                                 connection_parameter_groups):
            self._sources.update(presynaptic_cells.tolist())
            try:
                weights = connection_parameter_group.pop('weight')
                delays = connection_parameter_group.pop('delay')
                # nest.Connect expects a 2D array
                if not np.isscalar(weights):
                    weights = np.array([weights])
                if not np.isscalar(delays):
                    delays = np.array([delays])
                syn_dict.update({'weight': weights, 'delay': delays})

                if postsynaptic_cell.celltype.standard_receptor_type:
                    # For Tsodyks-Markram synapses models we set the "tau_psc" parameter to match
                    # the relevant "tau_syn" parameter from the post-synaptic neuron.
                    if 'tsodyks' in self.nest_synapse_model:
                        translations = postsynaptic_cell.celltype.translations
                        if self.receptor_type == 'inhibitory':
                            param_name = translations['tau_syn_I']['translated_name']
                        elif self.receptor_type == 'excitatory':
                            param_name = translations['tau_syn_E']['translated_name']
                        else:
                            raise NotImplementedError()
                        syn_dict["tau_psc"] = nest.GetStatus(postsynaptic_cell.node_collection,
                                                             param_name)[0]
                else:
                    syn_dict.update(
                        {"receptor_type": postsynaptic_cell.celltype.get_receptor_type(
                            self.receptor_type)})

                # For parameters other than weight and delay, we need to know if they are "common"
                # parameters (the same for all synapses) or "local" (different synapses can have
                # different values), as this affects how they are set.
                #
                # To introspect which parameters are common, we need an existing connection, so
                # the first time we create connections we pass just the weight and delay, and set
                # the other parameters later. We then get the list of common parameters and cache
                # it so that in subsequent Connect() calls we can pass all of the local
                # (non-common) parameters.

                if self._common_synapse_property_names is None:
                    nest.Connect(presynaptic_cells,
                                 postsynaptic_cell.node_collection,
                                 'all_to_all',
                                 syn_dict)
                    self._identify_common_synapse_properties()

                    # Retrieve connections so that we can set additional
                    # parameters using nest.SetStatus
                    connections = nest.GetConnections(source=presynaptic_cells,
                                                      target=postsynaptic_cell.node_collection,
                                                      synapse_model=self.nest_synapse_model,
                                                      synapse_label=self.nest_synapse_label)
                    for name, value in connection_parameter_group.items():
                        if name not in self._common_synapse_property_names:
                            value = make_sli_compatible(value)
                            if isinstance(value, np.ndarray):
                                nest.SetStatus(connections, name, value.tolist())
                            else:
                                nest.SetStatus(connections, name, value)
                        else:
                            self._set_common_synapse_property(name, value)
                else:
                    # Since we know which parameters are common, we can set the non-common
                    # parameters directly in the nest.Connect call
                    syn_dict = self._update_syn_params(syn_dict, connection_parameter_group)
                    nest.Connect(presynaptic_cells,
                                 postsynaptic_cell.node_collection,
                                 'all_to_all',
                                 syn_dict)
                    # and then set the common parameters
                    for name, value in connection_parameter_group.items():
                        if name in self._common_synapse_property_names:
                            self._set_common_synapse_property(name, value)

            except nest.NESTError as err:
                err_msg = (
                    f"{err}. presynaptic_cells={presynaptic_cells}, "
                    f"postsynaptic_cell={postsynaptic_cell}, "
                    f"weights={weights}, delays={delays}, "
                    f"synapse model='{self.nest_synapse_model}'"
                )
                raise errors.ConnectionError(err_msg)

        # Reset the caching of the connection list, since this will have to be recalculated
        self._connections = None
        self._simulator.state.stale_connection_cache = True

    def _set_attributes(self, parameter_space):
        if (
            "tau_minus" in parameter_space.keys()
            and not parameter_space["tau_minus"].is_homogeneous
        ):
            raise ValueError("tau_minus cannot be heterogeneous "
                             "within a single Projection with NEST.")
        # only columns for connections that exist on this machine
        parameter_space.evaluate(mask=(slice(None), self.post._mask_local))
        sources = nest.NodeCollection(sorted(self._sources))
        if self._common_synapse_property_names is None:
            self._identify_common_synapse_properties()
        for postsynaptic_cell, connection_parameters in zip(self.post.local_cells,
                                                            parameter_space.columns()):
            connections = nest.GetConnections(source=sources,
                                              target=postsynaptic_cell.node_collection,
                                              synapse_model=self.nest_synapse_model,
                                              synapse_label=self.nest_synapse_label)
            if connections:
                source_mask = self.pre.id_to_index(list(connections.sources()))
                for name, value in connection_parameters.items():
                    if (
                        name == "weight"
                        and self.receptor_type == 'inhibitory'
                        and self.post.conductance_based
                    ):
                        # NEST uses negative values for inhibitory weights,
                        # even if these are conductances
                        value *= -1
                    if name == "tau_minus":  # set on the post-synaptic cell
                        nest.SetStatus(self.post.node_collection[self.post.node_collection.local],
                                       {"tau_minus": simplify(value)})
                    elif name not in self._common_synapse_property_names:
                        value = make_sli_compatible(value)
                        if len(source_mask) > 1:
                            nest.SetStatus(connections, name, value[source_mask])
                        elif isinstance(value, np.ndarray):  # OneToOneConnector
                            nest.SetStatus(connections, name, value[source_mask])
                        else:
                            nest.SetStatus(connections, name, value)
                    else:
                        self._set_common_synapse_property(name, value)

    def _set_common_synapse_property(self, name, value):
        """
            Sets the common synapse property while making sure its value stays
            unique (i.e. it can only be set once).
        """
        if name in self._common_synapse_properties:
            unequal = self._common_synapse_properties[name] != value
            # handle both scalars and numpy ndarray
            if isinstance(unequal, np.ndarray):
                raise_error = unequal.any()
            else:
                raise_error = unequal
            if raise_error:
                raise ValueError("{} cannot be heterogeneous "
                                 "within a single Projection. Warning: "
                                 "Projection was only partially initialized."
                                 " Please call sim.nest.reset() to reset "
                                 "your network and start over!".format(name))
        if hasattr(value, "__len__"):
            value1 = value[0]
        else:
            value1 = value
        self._common_synapse_properties[name] = value1
        # we delay make_sli_compatible until this late stage so that we can
        # distinguish "parameter is an array consisting of scalar values"
        # (one value per connection) from
        # "parameter is a scalar value containing an array"
        # (one value for the entire projection)
        # In the latter case the value is wrapped in a Sequence object,
        # which is removed by make_sli_compatible
        value2 = make_sli_compatible(value1)
        nest.SetDefaults(self.nest_synapse_model, name, value2)

    # def saveConnections(self, file, gather=True, compatible_output=True):
    #    """
    #    Save connections to file in a format suitable for reading in with a
    #    FromFileConnector.
    #    """
    #    import operator
    #
    #    if isinstance(file, str):
    #        file = recording.files.StandardTextFile(file, mode='w')
    #
    #    lines   = nest.GetStatus(self.nest_connections, ('source', 'target', 'weight', 'delay'))
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
    #        lines       = np.array(lines, dtype=float)
    #        lines[:,2] *= 0.001
    #        if compatible_output:
    #            lines[:,0] = self.pre.id_to_index(lines[:,0])
    #            lines[:,1] = self.post.id_to_index(lines[:,1])
    #        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
    #        file.close()

    def _get_attributes_as_list(self, names):
        nest_names = []
        for name in names:
            if name == 'presynaptic_index':
                nest_names.append('source')
            elif name == 'postsynaptic_index':
                nest_names.append('target')
            else:
                nest_names.append(name)
        values = nest.GetStatus(self.nest_connections, nest_names)
        values = np.array(values)  # ought to preserve int type for source, target
        if 'weight' in names:
            # other attributes could also have scale factors - need to use translation mechanisms
            scale_factors = np.ones(len(names))
            scale_factors[names.index('weight')] = 0.001
            if self.receptor_type == 'inhibitory' and self.post.conductance_based:
                # NEST uses negative values for inhibitory weights, even if these are conductances
                scale_factors[names.index('weight')] *= -1
            values *= scale_factors
        if 'presynaptic_index' in names:
            values[:, names.index('presynaptic_index')] = self.pre.id_to_index(
                values[:, names.index('presynaptic_index')])
        if 'postsynaptic_index' in names:
            values[:, names.index('postsynaptic_index')] = self.post.id_to_index(
                values[:, names.index('postsynaptic_index')])
        values = values.tolist()
        for i in range(len(values)):
            values[i] = tuple(values[i])
        return values

    def _get_attributes_as_arrays(self, names, multiple_synapses='sum'):
        multi_synapse_operation = Projection.MULTI_SYNAPSE_OPERATIONS[multiple_synapses]
        all_values = []
        for attribute_name in names:
            if attribute_name[-1] == "s":  # weights --> weight, delays --> delay
                attribute_name = attribute_name[:-1]
            value_arr = np.nan * np.ones((self.pre.size, self.post.size))
            connection_attributes = nest.GetStatus(self.nest_connections,
                                                   ('source', 'target', attribute_name))
            for conn in connection_attributes:
                # (offset is always 0,0 for connections created with connect())
                src, tgt, value = conn
                addr = self.pre.id_to_index(src), self.post.id_to_index(tgt)
                if np.isnan(value_arr[addr]):
                    value_arr[addr] = value
                else:
                    value_arr[addr] = multi_synapse_operation(value_arr[addr], value)
            if attribute_name == 'weight':
                value_arr *= 0.001
                if self.receptor_type == 'inhibitory' and self.post.conductance_based:
                    # NEST uses negative values for inhibitory weights,
                    # even if these are conductances
                    value_arr *= -1
            all_values.append(value_arr)
        return all_values

    def _set_initial_value_array(self, variable, value):
        local_value = value.evaluate(simplify=True)
        nest.SetStatus(self.nest_connections, variable, local_value)
