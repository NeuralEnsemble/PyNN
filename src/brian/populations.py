"""

"""

import numpy
from brian import mV
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from . import simulator
from .recording import Recorder



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
            value = simplify(getattr(self.brian_group, name))
            if isinstance(value, numpy.ndarray):
                value = value[self.mask]
            parameter_dict[name] = value
        return ParameterSpace(parameter_dict, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            if name == "spike_times":
                self.brian_group._set_spike_times(value, self.mask)    
            else:
                getattr(self.brian_group, name)[self.mask] = value

    def _set_initial_value_array(self, variable, initial_values):
        raise NotImplementedError

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)
    
    @property
    def brian_group(self):
        return self.parent.brian_group


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        id_range = numpy.arange(simulator.state.id_counter,
                                simulator.state.id_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)
        self._mask_local = numpy.ones((self.size,), bool) # all cells are local. This doesn't seem very efficient.
        
        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        parameter_space.evaluate(simplify=False)
        
        #print(self.celltype.eqs)
        #print(list(parameter_space.items()))
        self.brian_group = self.celltype.brian_model(self.size,
                                                     self.celltype.eqs,
                                                     **parameter_space)
        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        simulator.state.network.add(self.brian_group)
    
    def _set_initial_value_array(self, variable, value):
        D = self.celltype.state_variable_translations[variable]
        pname = D['translated_name']
        pval = eval(D['forward_transform'], globals(), {variable: value})
        pval = pval.evaluate(simplify=False)
        self.brian_group.initial_values[pname] = pval
        self.brian_group.initialize()

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = simplify(getattr(self.brian_group, name))
            parameter_dict[name] = value
        return ParameterSpace(parameter_dict, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            setattr(self.brian_group, name, value)
