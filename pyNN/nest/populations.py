# -*- coding: utf-8 -*-
"""
NEST v3 implementation of the PyNN API.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
from collections import defaultdict

import numpy as np
import nest

from .. import common, errors
from ..parameters import ArrayParameter, Sequence, ParameterSpace, simplify, LazyArray
from ..random import RandomDistribution
from ..standardmodels import StandardCellType
from . import simulator
from .recording import Recorder

logger = logging.getLogger("PyNN")


class PopulationMixin(object):

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _set_parameters(self, parameter_space):
        """
        parameter_space should contain native parameters
        """
        param_dict = _build_params(parameter_space, np.where(self._mask_local)[0])
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = self.node_collection_source[self._mask_local]
        else:
            ids = self.node_collection[self._mask_local]
        simulator.state.set_status(ids, param_dict)

    def _get_parameters(self, *names):
        """
        Return a ParameterSpace containing PyNN parameters

        `names` should be PyNN names
        """
        def _get_component_parameters(component, names, component_label=None):
            if component.computed_parameters_include(names):
                # need all parameters in order to calculate values
                native_names = component.get_native_names()
            else:
                native_names = component.get_native_names(*names)
            native_parameter_space = self._get_native_parameters(*native_names)
            ps = component.reverse_translate(native_parameter_space)
            # extract values for this component from any ArrayParameters
            for name, value in ps.items():
                if isinstance(value.base_value, ArrayParameter):
                    index = self.celltype.receptor_types.index(component_label)
                    ps[name] = LazyArray(value.base_value[index])
                    ps[name].operations = value.operations
            return ps

        if isinstance(self.celltype, StandardCellType):
            if any("." in name for name in names):
                names_by_component = defaultdict(list)
                for name in names:
                    parts = name.split(".")
                    if len(parts) == 1:
                        names_by_component["neuron"].append(parts[0])
                    elif len(parts) == 2:
                        names_by_component[parts[0]].append(parts[1])
                    else:
                        raise ValueError("Invalid name: {}".format(name))
                if "neuron" in names_by_component:
                    parameter_space = _get_component_parameters(self.celltype.neuron,
                                                                names_by_component.pop("neuron"))
                else:
                    parameter_space = ParameterSpace({})
                for component_label, names in names_by_component.items():
                    parameter_space.add_child(
                        component_label,
                        _get_component_parameters(
                            self.celltype.post_synaptic_receptors[component_label],
                            names_by_component[component_label],
                            component_label))
            else:
                parameter_space = _get_component_parameters(self.celltype, names)
        else:
            parameter_space = self._get_native_parameters(*names)
        return parameter_space

    def _get_native_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters

        `names` should be native NEST names
        """
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = self.node_collection_source[self._mask_local]
        else:
            ids = self.node_collection[self._mask_local]

        if "spike_times" in names:
            parameter_dict = {"spike_times": [Sequence(value)
                                              for value in nest.GetStatus(ids, names)]}
        else:
            parameter_dict = {}
            for name in names:  # one name at a time, since some parameter values may be tuples
                val = np.array(nest.GetStatus(ids, name))
                if isinstance(val[0], tuple) or len(val.shape) == 2:
                    val = np.array([ArrayParameter(v) for v in val])
                    val = LazyArray(simplify(val), shape=(self.local_size,), dtype=ArrayParameter)
                    parameter_dict[name] = val
                else:
                    parameter_dict[name] = simplify(val)
        ps = ParameterSpace(parameter_dict, shape=(self.local_size,))
        return ps

    @property
    def local_node_collection(self):
        return self.node_collection[self._mask_local]


class Assembly(common.Assembly):
    __doc__ = common.Assembly.__doc__
    _simulator = simulator

    @property
    def local_node_collection(self):
        result = self.populations[0].local_node_collection
        for p in self.populations[1:]:
            result += p.local_node_collection
        return result

    @property
    def node_collection(self):
        return sum((p.node_collection for p in self.populations[1:]),
                   start=self.populations[0].node_collection)


class PopulationView(common.PopulationView, PopulationMixin):
    __doc__ = common.PopulationView.__doc__
    _simulator = simulator
    _assembly_class = Assembly

    @property
    def node_collection(self):
        return self.parent.node_collection[self.mask]

    @property
    def node_collection_source(self):
        return self.parent.node_collection_source[self.mask]


def _build_params(parameter_space, mask_local, size=None, extra_parameters=None):
    """
    Return either a single parameter dict or a list of dicts, suitable for use
    in Create or SetStatus.
    """
    if "UNSUPPORTED" in parameter_space.keys():
        parameter_space.pop("UNSUPPORTED")
    if size:
        parameter_space.shape = (size,)
    if parameter_space.is_homogeneous:
        parameter_space.evaluate(simplify=True)
        cell_parameters = parameter_space.as_dict()
        if extra_parameters:
            cell_parameters.update(extra_parameters)
        for name, val in cell_parameters.items():
            if isinstance(val, ArrayParameter):
                cell_parameters[name] = val.value.tolist()
    else:
        parameter_space.evaluate(mask=mask_local)
        cell_parameters = list(parameter_space)  # may not be the most efficient way.
        # Might be best to set homogeneous parameters on creation,
        # then inhomogeneous ones using SetStatus. Need some timings.
        for D in cell_parameters:
            for name, val in D.items():
                if isinstance(val, ArrayParameter):
                    D[name] = val.value.tolist()
            if extra_parameters:
                D.update(extra_parameters)
    return cell_parameters


class Population(common.Population, PopulationMixin):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        self._deferred_parrot_connections = False
        super(Population, self).__init__(size, cellclass,
                                         cellparams, structure, initial_values, label)
        self._simulator.state.populations.append(self)

    def _create_cells(self):
        """
        Create cells in NEST using the celltype of the current Population.
        """
        # this method should never be called more than once
        # perhaps should check for that
        nest_model = self.celltype.nest_name[simulator.state.spike_precision]
        if isinstance(self.celltype, StandardCellType):
            self.celltype.parameter_space.shape = (self.size,)  # should perhaps do this on a copy?
            params = _build_params(self.celltype.native_parameters,
                                   None,
                                   size=self.size,
                                   extra_parameters=self.celltype.extra_parameters)
        else:
            params = _build_params(self.celltype.parameter_space,
                                   None,
                                   size=self.size)
        try:
            self.node_collection = nest.Create(nest_model, self.size, params=params)
        except nest.NESTError as err:
            if "UnknownModelName" in err.args[0] and "cond" in err.args[0]:
                raise errors.InvalidModelError(
                    f"{err} Have you compiled NEST with the GSL (Gnu Scientific Library)?")
            if "Spike times must be sorted in non-descending order" in err.args[0]:
                raise errors.InvalidParameterValueError(
                    "Spike times given to SpikeSourceArray must be in increasing order")
            raise  # errors.InvalidModelError(err)
        # create parrot neurons if necessary
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            # we put the parrots into all_cells, since this will
            # be used for connections and recording. all_cells_source
            # should be used for setting parameters
            self.node_collection_source = self.node_collection
            parrot_model = (
                simulator.state.spike_precision == "off_grid"
                and "parrot_neuron_ps"
                or "parrot_neuron"
            )
            self.node_collection = nest.Create(parrot_model, self.size)

            self._deferred_parrot_connections = True
            # connecting up the parrot neurons is deferred until we know the value of min_delay
            # which could be 'auto' at this point.
        if self.node_collection.local is True:
            self._mask_local = np.array([True])
        else:
            self._mask_local = np.array(self.node_collection.local)
        self.all_cells = np.array([simulator.ID(gid) for gid in self.node_collection.tolist()],
                                  simulator.ID)
        for gid in self.all_cells:
            gid.parent = self
            gid.node_collection = nest.NodeCollection([int(gid)])
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            for gid, source in zip(self.all_cells, self.node_collection_source.tolist()):
                gid.source = source

    def _connect_parrot_neurons(self):
        nest.Connect(self.node_collection_source, self.node_collection, 'one_to_one',
                     syn_spec={'delay': simulator.state.min_delay})
        self._deferred_parrot_connections = False

    def _reset(self):
        # adjust parameters that represent absolute times for the time offset after reset
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            for name in self.celltype.get_native_names():
                if name in ("start", "stop", "spike_times"):
                    value = self.celltype.native_parameters[name]
                    self._simulator.set_status(self.node_collection, name, value)

    def _set_initial_value_array(self, variable, value):
        if hasattr(self.celltype, "variable_map"):
            variable = self.celltype.variable_map[variable]
        if isinstance(value.base_value, RandomDistribution) and value.base_value.rng.parallel_safe:
            local_values = value.evaluate()[self._mask_local]
        else:
            local_values = value._partially_evaluate(self._mask_local, simplify=True)
        try:
            if (
                self._mask_local.dtype == bool
                and self._mask_local.size == 1
                and self._mask_local[0]
            ):
                simulator.state.set_status(self.node_collection, variable, local_values)
            else:
                simulator.state.set_status(self.node_collection[self._mask_local],
                                           variable, local_values)
        except nest.NESTError as e:
            if "Unused dictionary items" in e.args[0]:
                logger.warning("NEST does not allow setting an initial value for %s" % variable)
                # should perhaps check whether value-to-be-set is the same as current value,
                # and raise an Exception if not, rather than just emit a warning.
            else:
                raise
