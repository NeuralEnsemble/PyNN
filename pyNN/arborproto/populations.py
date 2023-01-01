# encoding: utf-8
"""
Arbor implementation of the PyNN API.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy as np
from pyNN import common, errors
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from pyNN.arborproto import simulator
from pyNN.arborproto.recording import Recorder
from pyNN.arborproto.standardmodels.cells import MultiCompartmentNeuron, SpikeSourcePoisson
import arbor

import logging

logger = logging.getLogger("PyNN")


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,))  # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        for name, value in parameter_space.items():
            try:
                self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)
            except ValueError as err:
                raise errors.InvalidParameterValueError(f"{name} should not be of type {type(value)}")

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        id_range = np.arange(simulator.state.gid_counter,
                             simulator.state.gid_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)

        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank

        self._mask_local = is_local(self.all_cells)

        parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        # parameter_space.evaluate(mask=self._mask_local, simplify=False)
        parameter_space.evaluate(mask=None, simplify=True)
        self._parameters = parameter_space.as_dict()
        self.parameters = parameter_space

        for id in self.all_cells:
            id.parent = self
        simulator.state.gid_counter += self.size

        for i, (id, is_local, params) in enumerate(zip(self.all_cells, self._mask_local, parameter_space)):
            self.all_cells[i] = simulator.ID(id)
            self.all_cells[i].parent = self
            if is_local:
                if hasattr(self.celltype, "extra_parameters"):
                    params.update(self.celltype.extra_parameters)
                if isinstance(self.celltype, MultiCompartmentNeuron):
                    self.all_cells[i]._build_cell(self.celltype.model, params)
                if isinstance(self.celltype, SpikeSourcePoisson):
                    #self.all_cells[i]._build_cell(self.celltype.model, params)
                    self.all_cells[i]._cell = self.celltype.model(params)

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = simplify(self._parameters[name])
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False, mask=self._mask_local)
        for name, value in parameter_space.items():
            self._parameters[name] = value
