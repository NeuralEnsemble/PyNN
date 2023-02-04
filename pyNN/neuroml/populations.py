"""

Export of PyNN models to NeuroML 2

Contact Padraig Gleeson for more details

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa

import numpy as np
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from . import simulator
from .recording import Recorder
import logging

import neuroml


logger = logging.getLogger("PyNN_NeuroML")

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
        return ParameterSpace(parameter_dict, shape=(self.size,)) # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        #ps = self.parent._get_parameters(*self.celltype.get_native_names())
        for name, value in parameter_space.items():
            self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)
            #ps[name][self.mask] = value.evaluate(simplify=True)
        #ps.evaluate(simplify=True)
        #self.parent._parameters = ps.as_dict()

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)



class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        super(Population, self).__init__(size, cellclass, cellparams, structure,initial_values, label)

        logger.debug("Created NeuroML Population: %s of size %i" % (self.label, self.size))

        for cell in self.all_cells:
            index = self.id_to_index(cell)
            inst = neuroml.Instance(id=index)
            self.pop.instances.append(inst)
            x = self.positions[0][index]
            y = self.positions[1][index]
            z = self.positions[2][index]
            logger.debug("Creating cell at (%s, %s, %s)"%(x,y,z))
            inst.location = neuroml.Location(x=x,y=y,z=z)


    def _create_cells(self):
        """Create the cells in the population"""

        nml_doc = simulator._get_nml_doc()
        net = simulator._get_main_network()

        cell_pynn = self.celltype.__class__.__name__
        logger.debug("Creating Cell instance: %s" % (cell_pynn))

        cell_id = self.celltype.add_to_nml_doc(nml_doc, self)

        logger.debug("Creating Population: %s of size %i" % (self.label, self.size))

        self.pop = neuroml.Population(id=self.label, size = self.size, type="populationList",
                          component=cell_id)
        net.populations.append(self.pop)


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
        parameter_space.evaluate(mask=self._mask_local, simplify=False)
        self._parameters = parameter_space.as_dict()

        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size


    def annotate(self, **annotations):
        print("Updating annotations: %s"%annotations)
        for k in annotations:

            self.pop.properties.append(neuroml.Property(k, annotations[k]))

        self.annotations.update(annotations)

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
        #ps = self._get_parameters(*self.celltype.get_native_names())
        #ps.update(**parameter_space)
        #ps.evaluate(simplify=True)
        #self._parameters = ps.as_dict()
        parameter_space.evaluate(simplify=False, mask=self._mask_local)
        for name, value in parameter_space.items():
            self._parameters[name] = value
