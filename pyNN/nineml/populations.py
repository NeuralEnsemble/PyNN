"""
Export of PyNN scripts as NineML.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy as np
import nineml.user as nineml

from pyNN import common
from pyNN.standardmodels import StandardCellType
from . import simulator
from .recording import Recorder
from .utility import build_parameter_set, catalog_url


class BasePopulation(object):

    def get_synaptic_response_components(self, synaptic_mechanism_name):
        return [self.celltype.synaptic_receptor_component_to_nineml(
            synaptic_mechanism_name, self.label, (self.size,))]

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Assembly(common.Assembly):
    _simulator = simulator

    def __init__(self, label=None, *populations):
        common.Assembly.__init__(self, label, *populations)
        self._simulator.state.net.assemblies.append(self)

    def get_synaptic_response_components(self, synaptic_mechanism_name):
        components = set([])
        for p in self.populations:
            components.add(p.celltype.synaptic_receptor_component_to_nineml(
                synaptic_mechanism_name, self.label))
        return components

    def to_nineml(self):
        group = nineml.Group(self.label)
        for p in self.populations:
            group.add(p.to_nineml())
        return group


class PopulationView(BasePopulation, common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def __init__(self, parent, selector, label=None):
        common.PopulationView.__init__(self, parent, selector, label)
        self._simulator.state.net.populations.append(self)

    def to_nineml(self):
        if isinstance(self.mask, slice):
            ids = "%s:%s:%s" % (self.mask.start or "", self.mask.stop or "", self.mask.step or "")
        else:
            ids = str(self.mask.tolist())
        selection = nineml.Selection(self.label,
                                     nineml.All(
                                         nineml.Eq("population[@name]", self.parent.label),
                                         nineml.In("population[@id]", ids)
                                     )
                                     )
        return selection


class Population(BasePopulation, common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        common.Population.__init__(self, size, cellclass, cellparams, structure,
                                   initial_values, label)
        self._simulator.state.net.populations.append(self)

    def _create_cells(self):
        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)

        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)

        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        self._parameters = parameter_space

        for id in self.all_cells:
            id.parent = self
        self._simulator.state.id_counter += self.size

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        self._parameters.update(parameter_space)

    def to_nineml(self):
        if self.structure:
            structure = nineml.Structure(
                name="structure for %s" % self.label,
                definition=nineml.Definition(
                    f"{catalog_url}/networkstructures/{self.structure.__class__.__name__}.xml",
                    "structure"),
                parameters=build_parameter_set(self.structure.get_parameters())
            )
        else:
            structure = None
        population = nineml.Population(
            name=self.label,
            number=len(self),
            prototype=self.celltype.to_nineml(self.label, (self.size,))[0],
            positions=nineml.PositionList(structure=structure))
        return population
