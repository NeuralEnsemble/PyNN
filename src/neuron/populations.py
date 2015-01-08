# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import logging
from pyNN import common
from pyNN.parameters import Sequence, ParameterSpace, simplify
from pyNN.standardmodels import StandardCellType
from pyNN.random import RandomDistribution
from . import simulator
from .recording import Recorder

logger = logging.getLogger("PyNN")


class PopulationMixin(object):

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        parameter_space.evaluate(mask=numpy.where(self._mask_local)[0])
        for cell, parameters in zip(self, parameter_space):
            for name, val in parameters.items():
                setattr(cell._cell, name, val)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            if name == 'spike_times': # hack
                parameter_dict[name] = [Sequence(getattr(id._cell, name)) for id in self]
            else:
                parameter_dict[name] = simplify(numpy.array([getattr(id._cell, name) for id in self]))
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_initial_value_array(self, variable, initial_values):
        if initial_values.is_homogeneous:
            value = initial_values.evaluate(simplify=True)
            for cell in self:  # only on local node
                setattr(cell._cell, "%s_init" % variable, value)
        else:
            if isinstance(initial_values.base_value, RandomDistribution) and initial_values.base_value.rng.parallel_safe:
                local_values = initial_values.evaluate()[self._mask_local]
            else:
                local_values = initial_values[self._mask_local]            
            for cell, value in zip(self, local_values):
                setattr(cell._cell, "%s_init" % variable, value)


class Assembly(common.Assembly):
    __doc__ = common.Assembly.__doc__
    _simulator = simulator


class PopulationView(common.PopulationView, PopulationMixin):
    __doc__ = common.PopulationView.__doc__
    _simulator = simulator
    _assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population, PopulationMixin):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        common.Population.__init__(self, size, cellclass, cellparams,
                                   structure, initial_values, label)
        simulator.initializer.register(self)

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _create_cells(self):
        """
        Create cells in NEURON using the celltype of the current Population.
        """
        # this method should never be called more than once
        # perhaps should check for that
        self.first_id = simulator.state.gid_counter
        self.last_id = simulator.state.gid_counter + self.size - 1
        self.all_cells = numpy.array([id for id in range(self.first_id, self.last_id+1)], 
                                     simulator.ID)
        # mask_local is used to extract those elements from arrays that apply to the cells on the current node
        self._mask_local = self.all_cells%simulator.state.num_processes==simulator.state.mpi_rank # round-robin distribution of cells between nodes
        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        parameter_space.evaluate(mask=None)
        for i, (id, is_local, params) in enumerate(zip(self.all_cells, self._mask_local, parameter_space)):
            self.all_cells[i] = simulator.ID(id)
            self.all_cells[i].parent = self
            if is_local:
                if hasattr(self.celltype, "extra_parameters"):
                    params.update(self.celltype.extra_parameters)
                self.all_cells[i]._build_cell(self.celltype.model, params)
        simulator.initializer.register(*self.all_cells[self._mask_local])
        simulator.state.gid_counter += self.size

    def _native_rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        assert isinstance(rand_distr.rng, NativeRNG)
        rng = simulator.h.Random(rand_distr.rng.seed or 0)
        native_rand_distr = getattr(rng, rand_distr.name)
        rarr = [native_rand_distr(*rand_distr.parameters)] + [rng.repick() for i in range(self.all_cells.size-1)]
        self.tset(parametername, rarr)
