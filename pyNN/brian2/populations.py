"""
Brian 2 implementation of Population, PopulationView and Assembly.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
import numpy as np
from .. import common
from ..standardmodels import StandardCellType
from ..parameters import ArrayParameter, ParameterSpace, simplify, LazyArray
from . import simulator
from .recording import Recorder
import brian2
ms = brian2.ms
mV = brian2.mV


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationMixin(object):

    def _get_parameters(self, *names):
        """
        Return a ParameterSpace containing PyNN parameters

        `names` should be PyNN names
        """
        def _get_component_parameters(component, names, component_label=None):
            kwargs = {}
            if component_label:
                kwargs["suffix"] = component_label
            if component.computed_parameters_include(names):
                # need all parameters in order to calculate values
                native_names = component.get_native_names(**kwargs)
            else:
                native_names = component.get_native_names(*names, **kwargs)
            native_parameter_space = self._get_native_parameters(*native_names)
            if component_label:
                ps = component.reverse_translate(native_parameter_space, suffix=component_label)
            else:
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


class PopulationView(common.PopulationView, PopulationMixin):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        if isinstance(self.celltype, StandardCellType):
            if any(name in self.celltype.computed_parameters() for name in names):
                # need all parameters in order to calculate values
                native_names = self.celltype.get_native_names()
            else:
                native_names = self.celltype.get_native_names(*names)
            native_parameter_space = self._get_native_parameters(*native_names)
            parameter_space = self.celltype.reverse_translate(native_parameter_space)
        else:
            parameter_space = self._get_native_parameters(*native_names)
        return parameter_space

    def _get_native_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = getattr(self.brian2_group, name)
            if hasattr(value, "shape") and value.shape:
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            if name == "spike_time_sequences":
                self.brian2_group._set_spike_time_sequences(value, self.mask)
            elif name == "tau_refrac":  # cannot be heterogeneous
                self.tau_refrac = value
            else:
                getattr(self.brian2_group, name)[self.mask] = value

    def _set_initial_value_array(self, variable, initial_values):
        raise NotImplementedError

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    @property
    def brian2_group(self):
        return self.parent.brian2_group


class Population(common.Population, PopulationMixin):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)
        # all cells are local. This doesn't seem very efficient.
        self._mask_local = np.ones((self.size,), bool)

        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space

        parameter_space.shape = (self.size,)
        parameter_space.evaluate(simplify=False)
        parameter_space.flatten(with_prefix=False)
        self.brian2_group = self.celltype.brian2_model(self.size,
                                                       self.celltype.eqs,
                                                       **parameter_space)
        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        simulator.state.network.add(self.brian2_group)

    def _set_initial_value_array(self, variable, value):
        D = self.celltype.state_variable_translations[variable]
        pname = D['translated_name']
        if callable(D['forward_transform']):
            pval = D['forward_transform'](**{variable: value})
        else:
            pval = eval(D['forward_transform'], globals(), {variable: value})
        pval = pval.evaluate(simplify=False)
        self.brian2_group.initial_values[pname] = pval
        self.brian2_group.initialize()

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_native_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = getattr(self.brian2_group, name)
            if hasattr(value, "shape") and value.shape != ():
                value = value[:]
            parameter_dict[name] = value
        return ParameterSpace(parameter_dict, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            if (name == "tau_refrac"):
                value = simplify(value)
                self.brian2_group.tau_refrac = value
            elif (name == "v_reset"):
                value = simplify(value)
                self.brian2_group.v_reset = value
            else:
                setattr(self.brian2_group, name, value)
