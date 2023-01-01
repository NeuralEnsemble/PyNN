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
from pyNN.arbor import simulator
from pyNN.arbor.recording import Recorder
from pyNN.arbor.standardmodels.cells import MultiCompartmentNeuron, SpikeSourcePoisson
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

    # def __init__(self, size, cellclass, cellparams=None, structure=None,
    #              initial_values={}, label=None):
    #     # Needed to avoid TypeError: type object argument after ** must be a mapping, not NoneType
    #     # when calling self.celltype = cellclass(**cellparams)
    #     __doc__ = common.Population.__doc__
    #     common.Population.__init__(self, size, cellclass, cellparams,
    #                                structure, initial_values, label)
    #     # simulator.initializer.register(self)

    def _create_cells(self):
        """
        Create cells in NEURON using the celltype of the current Population.
        """
        # this method should never be called more than once
        # perhaps should check for that
        self.first_id = simulator.state.gid_counter
        self.last_id = simulator.state.gid_counter + self.size - 1
        self.all_cells = np.array([id for id in range(self.first_id, self.last_id + 1)],
                                  simulator.ID)

        # mask_local is used to extract those elements from arrays that apply to the cells on the current node
        # round-robin distribution of cells between nodes
        self._mask_local = self.all_cells % simulator.state.num_processes == simulator.state.mpi_rank

        # Issues with calling native_parameters:
        #   1 method of StandardModelType in pyNN/standardmodels/__init__.py
        #   2 self.parameter_space is the parameters for instantiating the cell type (e.g. MultiCompartmentNeuron)
        #   3 parameter_space.schema must be the same as celltype.get_schema() [parameters.schema != self.get_schema()]
        #   4 iterate the parameter_space by extracting the keys and get its corresponding value in translations
        #       * translations is the attribute defined under the cell type class
        #   5 because of (4) keys in parameter_space MUST BE the names defined in translations
        # print(self.celltype.parameter_space)
        # print(self.celltype.parameter_space.schema)
        # To uncomment the condition below FIX why calling native_parameters changes the celltype schema.
        # Notice that the output of the above print statement is the same as in ~/arbor/standardmodels/cell.py
        # if isinstance(self.celltype, StandardCellType):
        #     parameter_space = self.celltype.native_parameters
        # else:
        #     parameter_space = self.celltype.parameter_space
        parameter_space = self.celltype.parameter_space # temporary fix in place of the above condition
        # print("output of celltype.parameter_space")
        # print(self.celltype.parameter_space)
        # print("output of its schema")
        # import pprint
        # pprint.pprint(self.celltype.parameter_space.schema)
        # print("output of celltype.native_parameters")
        # print(self.celltype.native_parameters) # TODO: fix native_parameters function in pynn/parameters.py
        # print(parameter_space.schema)
        # print(parameter_space)
        # print(parameter_space["morphology"])
        # print(parameter_space["morphology"].base_value)
        # print(parameter_space["morphology"].base_value.item())
        # print(parameter_space["morphology"].base_value.item()._morphology)

        parameter_space.shape = (self.size,)
        parameter_space.evaluate(mask=None, simplify=True)
        for i, (id, is_local, params) in enumerate(zip(self.all_cells, self._mask_local, parameter_space)):
            self.all_cells[i] = simulator.ID(id)
            self.all_cells[i].parent = self
            # print(i)
            # print("population parameters")
            # print(params)
            if is_local:
                if hasattr(self.celltype, "extra_parameters"):
                    # print("extra_parameters")
                    # print(self.celltype.extra_parameters)
                    params.update(self.celltype.extra_parameters)
                    # print(params)
                    # print(params["morphology"])
                    # print(params["morphology"]._morphology) # something happened because ._morphology is not found
                if isinstance(self.celltype, MultiCompartmentNeuron):
                    self.all_cells[i]._build_cell(self.celltype.model, params)
        # simulator.initializer.register(*self.all_cells[self._mask_local])
        simulator.state.gid_counter += self.size

    def _set_initial_value_array(self, variable, initial_values):
        for i in range(self.all_cells.size):
            if variable == "v":
                # print(self.all_cells[i]._cell)
                # print(type(self.all_cells[i]._cell))
                # print(type(self.all_cells[i]._cell))
                # print(type(self.all_cells[i]._cell._decor))
                # print(initial_values)
                # for val in initial_values.evaluate(): # access lazy array alternative is initial_values.base_value
                #     self.all_cells[i]._cell._decor.set_property(Vm=val)
                # It is assumed that the array size of intial_values equals the self.all_cells.size (population size)
                self.all_cells[i]._cell._decor.set_property(Vm=initial_values.evaluate()[i])
        # for ky, value in initial_values.items():
        #     if ky == "v":
        #         self.all_cells[indx].__decor.set_property(Vm=value)
        #         for i in range(self.all_cells.size):
        #             self.all_cells[i].__decor.set_property(Vm = value)

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

    def record(self, record_type, location=None):  # Temporary placement, should be in Recorder class
        if record_type == "spikes":  # cells.record('spikes')
            # decor.place('"axon_terminal"', arbor.spike_detector(-10), "detector")
            for i in range(self.all_cells.size):
                self.all_cells[i]._cell._decor.place('"soma_midpoint"', arbor.spike_detector(-10), "detector")