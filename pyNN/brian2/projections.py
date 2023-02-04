"""
Brian 2 implementation of Projection

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from itertools import chain
from collections import defaultdict
import numpy as np
import brian2
from brian2 import uS, nA, mV, ms
from .. import common
from ..standardmodels.synapses import TsodyksMarkramSynapse
from ..core import is_listlike
from ..parameters import ParameterSpace
from ..space import Space
from . import simulator
from .standardmodels.synapses import StaticSynapse


logger = logging.getLogger("PyNN")


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, projection, i_group, j_group, index):
        self.projection = projection
        self.i_group = i_group
        self.j_group = j_group
        self.index = index
        self._syn_obj = self.projection._brian2_synapses[self.i_group][self.j_group]

    def _get(self, attr_name):
        value = getattr(self._syn_obj, attr_name)[self.index]
        native_ps = ParameterSpace({attr_name: value}, shape=(1,))
        ps = self.projection.synapse_type.reverse_translate(native_ps)
        ps.evaluate()
        return ps[attr_name]

    def _set(self, attr_name, value):
        ps = ParameterSpace({attr_name: value}, shape=(
            1,), schema=self.projection.synapse_type.get_schema())
        native_ps = self.projection.synapse_type.translate(ps)
        native_ps.evaluate()
        getattr(self._syn_obj, attr_name)[self.index] = native_ps[attr_name]

    def _set_weight(self, w):
        self._set("weight", w)

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return self._get("weight")

    def _set_delay(self, d):
        self._set("delay", d)

    def _get_delay(self):
        """Synaptic delay in ms."""
        return self._get("delay")

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        return tuple([getattr(self, name) for name in attribute_names])


def basic_units(units):
    # todo: implement this properly so it works with any units
    if units == mV:
        return "volt"
    if units == uS:
        return "siemens"
    if units == nA:
        return "ampere"
    raise Exception("Can't handle units '{}'".format(units))


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
        self._n_connections = 0
        # create one Synapses object per pre-post population pair
        # there will be multiple such pairs if either `presynaptic_population`
        # or `postsynaptic_population` is an Assembly.
        if isinstance(self.pre, common.Assembly):
            presynaptic_populations = self.pre.populations
        else:
            presynaptic_populations = [self.pre]
        if isinstance(self.post, common.Assembly):
            postsynaptic_populations = self.post.populations
            assert self.post._homogeneous_synapses, "Inhomogeneous assemblies not yet supported"
        else:
            postsynaptic_populations = [self.post]
        self._brian2_synapses = defaultdict(dict)
        for i, pre in enumerate(presynaptic_populations):
            for j, post in enumerate(postsynaptic_populations):
                # complete the synapse type equations according to the
                # post-synaptic response type
                psv = post.celltype.post_synaptic_variables[self.receptor_type]
                if (
                    hasattr(post.celltype, "voltage_based_synapses")
                    and post.celltype.voltage_based_synapses
                ):
                    weight_units = mV
                else:
                    weight_units = post.celltype.conductance_based and uS or nA
                self.synapse_type._set_target_type(weight_units)
                equation_context = {"syn_var": psv, "weight_units": basic_units(weight_units)}
                pre_eqns = self.synapse_type.pre % equation_context
                if self.synapse_type.post:
                    post_eqns = self.synapse_type.post % equation_context
                else:
                    post_eqns = None

                #  units are being transformed for exemple from amp to A
                model = self.synapse_type.eqs % equation_context

                # create the brian2 Synapses object.
                syn_obj = brian2.Synapses(pre.brian2_group, post.brian2_group,
                                          model=model, on_pre=pre_eqns,
                                          on_post=post_eqns,
                                          clock=simulator.state.network.clock,
                                          multisynaptic_index='synapse_number')
                # code_namespace={"exp": np.exp})
                self._brian2_synapses[i][j] = syn_obj
                simulator.state.network.add(syn_obj)
        # connect the populations
        connector.connect(self)
        # special-case: the Tsodyks-Markram short-term plasticity model takes
        #               a parameter value from the post-synaptic response model
        if isinstance(self.synapse_type, TsodyksMarkramSynapse):
            self._set_tau_syn_for_tsodyks_markram()

    def __len__(self):
        return self._n_connections

    @property
    def connections(self):
        """
        Returns an iterator over local connections in this projection, as `Connection` objects.
        """
        return (Connection(self, i_group, j_group, i)
                for i_group in range(len(self._brian2_synapses))
                for j_group in range(len(self._brian2_synapses[i_group]))
                for i in range(len(self._brian2_synapses[i_group][j_group]))
                )

    def _partition(self, indices):
        """
        partition indices, in case of Assemblies
        """
        if isinstance(self.pre, common.Assembly):
            boundaries = np.cumsum([0] + [p.size for p in self.pre.populations])
            assert indices.max() < boundaries[-1]
            partitions = np.split(indices, np.searchsorted(
                indices, boundaries[1:-1])) - boundaries[:-1]
            for i_group, local_indices in enumerate(partitions):
                if isinstance(self.pre.populations[i_group], common.PopulationView):
                    partitions[i_group] = self.pre.populations[i_group].index_in_grandparent(
                        local_indices)
        elif isinstance(self.pre, common.PopulationView):
            partitions = [self.pre.index_in_grandparent(indices)]
        else:
            partitions = [indices]
        return partitions

    def _localize_index(self, index):
        """determine which group the postsynaptic index belongs to """
        if isinstance(self.post, common.Assembly):
            boundaries = np.cumsum([0] + [p.size for p in self.post.populations])
            j = np.searchsorted(boundaries, index, side='right') - 1
            local_index = index - boundaries[j]
            if isinstance(self.post.populations[j], common.PopulationView):
                return j, self.post.populations[j].index_in_grandparent(local_index)
            else:
                return j, local_index
        elif isinstance(self.post, common.PopulationView):
            return 0, self.post.index_in_grandparent(index)
        else:
            return 0, index

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        connection_parameters.pop("dendritic_delay_fraction", None)  # TODO: need to to handle this
        presynaptic_index_partitions = self._partition(presynaptic_indices)
        j_group, j = self._localize_index(postsynaptic_index)
        # specify which connections exist
        for i_group, i in enumerate(presynaptic_index_partitions):
            if i.size > 0:
                self._brian2_synapses[i_group][j_group].connect(i=i, j=j)  # "[i, j]
                self._n_connections += i.size
        # set connection parameters

        for name, value in chain(connection_parameters.items(),
                                 self.synapse_type.initial_conditions.items()):
            if name == 'delay':
                scale = self._simulator.state.dt * ms
                value /= scale                         # ensure delays are rounded to the
                value = np.round(value) * scale     # nearest time step, rather than truncated
            for i_group, i in enumerate(presynaptic_index_partitions):
                if i.size > 0:
                    brian2_var = getattr(self._brian2_synapses[i_group][j_group], name)
                    if is_listlike(value):
                        for ii, v in zip(i, value):
                            brian2_var[ii, j] = v
                    else:
                        for ii in i:
                            try:
                                brian2_var[ii, j] = value
                            except TypeError as err:
                                if "read-only" in str(err):
                                    logger.info(
                                        f"Cannot set synaptic initial value for variable {name}")
                                else:
                                    raise
                    # brian2_var[i, j] = value
                    # ^ doesn't work with multiple connections between a given neuron pair.
                    #   Need to understand the internals of Synapses and SynapticVariable better

    def _set_attributes(self, connection_parameters):
        if isinstance(self.post, common.Assembly) or isinstance(self.pre, common.Assembly):
            raise NotImplementedError
        syn_obj = self._brian2_synapses[0][0]
        connection_parameters.evaluate()  # inefficient: would be better to evaluate using mask
        for name, value in connection_parameters.items():
            creation_order_sorted_value = value[syn_obj.i[:], syn_obj.j[:]]
            setattr(syn_obj, name, creation_order_sorted_value)

    def _get_attributes_as_arrays(self, attribute_names, multiple_synapses='sum'):
        if isinstance(self.post, common.Assembly) or isinstance(self.pre, common.Assembly):
            raise NotImplementedError
        values = []
        syn_obj = self._brian2_synapses[0][0]
        nan_mask = np.full((self.pre.size, self.post.size), True)
        iarr, jarr = syn_obj.i[:], syn_obj.j[:]
        nan_mask[iarr, jarr] = False

        multi_synapse_aggregation_map = {
            'sum': (np.add.at, 0.0),
            'min': (np.minimum.at, np.inf),
            'max': (np.maximum.at, -np.inf)
        }

        for name in attribute_names:
            value = getattr(syn_obj, name)[:]
            # should really use the translated name
            native_ps = ParameterSpace({name: value}, shape=value.shape)
            ps = self.synapse_type.reverse_translate(native_ps)
            ps.evaluate()

            if multiple_synapses in multi_synapse_aggregation_map:
                aggregation_func, dummy_val = multi_synapse_aggregation_map[multiple_synapses]
                array_val = np.full((self.pre.size, self.post.size), dummy_val)
                aggregation_func(array_val, (syn_obj.i[:], syn_obj.j[:]), ps[name])
                array_val[nan_mask] = np.nan
            else:
                raise NotImplementedError
            values.append(array_val)
        return values

    def _get_attributes_as_list(self, attribute_names):
        if isinstance(self.post, common.Assembly) or isinstance(self.pre, common.Assembly):
            raise NotImplementedError
        values = []
        syn_obj = self._brian2_synapses[0][0]
        for name in attribute_names:
            if name == "presynaptic_index":
                value = syn_obj.i[:]  # _indices.synaptic_pre.get_value()
                if hasattr(self.pre, "parent"):
                    # map index in parent onto index in view
                    value = self.pre.index_from_parent_index(value)
            elif name == "postsynaptic_index":
                value = syn_obj.j[:]  # _indices.synaptic_post.get_value()
                if hasattr(self.post, "parent"):
                    # map index in parent onto index in view
                    value = self.post.index_from_parent_index(value)
            else:
                value = getattr(syn_obj, name)[:]
                # should really use the translated name
                native_ps = ParameterSpace({name: value}, shape=value.shape)
                # todo: this whole "get attributes" thing needs refactoring
                #       in all backends to properly use translation
                ps = self.synapse_type.reverse_translate(native_ps)
                ps.evaluate()
                value = ps[name]
            values.append(value)
        a = np.array(values)

        return [tuple(x) for x in a.T]

    def _set_tau_syn_for_tsodyks_markram(self):
        if isinstance(self.post, common.Assembly) or isinstance(self.pre, common.Assembly):
            raise NotImplementedError
        tau_syn_var = self.synapse_type.tau_syn_var[self.receptor_type]
        self._brian2_synapses[0][0].tau_syn = self.post.get(
            tau_syn_var)[self._brian2_synapses[0][0].j] * brian2.ms
