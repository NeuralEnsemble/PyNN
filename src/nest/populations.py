# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import nest
import logging
from pyNN import common, errors
from pyNN.parameters import Sequence, ParameterSpace, simplify
from pyNN.standardmodels import StandardCellType
from . import simulator
from .recording import Recorder

STATE_VARIABLE_MAP = {"v": "V_m", "w": "w", "gsyn_exc": "g_ex",
                      "gsyn_inh": "g_in"}
logger = logging.getLogger("PyNN")


class PopulationMixin(object):

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _set_parameters(self, parameter_space):
        """
        parameter_space should contain native parameters
        """
        param_dict = _build_params(parameter_space, numpy.where(self._mask_local)[0])
        ids = self.local_cells.tolist()
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = [id.source for id in ids]
        nest.SetStatus(ids, param_dict)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        ids = self.local_cells.tolist()
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = [id.source for id in ids]
        parameter_array = numpy.array(nest.GetStatus(ids, names))
        parameter_dict = dict((name, simplify(parameter_array[:, col]))
                              for col, name in enumerate(names))
        if "spike_times" in parameter_dict: # hack
            parameter_dict["spike_times"] = [Sequence(value) for value in parameter_dict["spike_times"]]
        return ParameterSpace(parameter_dict, shape=(self.local_size,))


class Assembly(common.Assembly):
    __doc__ = common.Assembly.__doc__
    _simulator = simulator


class PopulationView(common.PopulationView, PopulationMixin):
    __doc__ = common.PopulationView.__doc__
    _simulator = simulator
    _assembly_class = Assembly


def _build_params(parameter_space, mask_local, size=None, extra_parameters=None):
    """
    Return either a single parameter dict or a list of dicts, suitable for use
    in Create or SetStatus.
    """
    if size:
        parameter_space.shape = (size,)
    if parameter_space.is_homogeneous:
        parameter_space.evaluate(simplify=True)
        cell_parameters = parameter_space.as_dict()
        if extra_parameters:
            cell_parameters.update(extra_parameters)
        for name, val in cell_parameters.items():
            if isinstance(val, Sequence):
                cell_parameters[name] = val.value
    else:
        parameter_space.evaluate(mask=mask_local)
        cell_parameters = list(parameter_space) # may not be the most efficient way. Might be best to set homogeneous parameters on creation, then inhomogeneous ones using SetStatus. Need some timings.
        for D in cell_parameters:
            for name, val in D.items():
                if isinstance(val, Sequence):
                    D[name] = val.value
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
        __doc__ = common.Population.__doc__
        super(Population, self).__init__(size, cellclass, cellparams, structure, initial_values, label)
        self._simulator.state.populations.append(self)

    def _create_cells(self):
        """
        Create cells in NEST using the celltype of the current Population.
        """
        # this method should never be called more than once
        # perhaps should check for that
        nest_model = self.celltype.nest_name[simulator.state.spike_precision]
        if isinstance(self.celltype, StandardCellType):
            params = _build_params(self.celltype.native_parameters,
                                   None,
                                   size=self.size,
                                   extra_parameters=self.celltype.extra_parameters)
        else:
            params = _build_params(self.celltype.parameter_space,
                                   None,
                                   size=self.size)
        try:
            self.all_cells = nest.Create(nest_model, self.size, params=params)
        except nest.NESTError, err:
            if "UnknownModelName" in err.message and "cond" in err.message:
                raise errors.InvalidModelError("%s Have you compiled NEST with the GSL (Gnu Scientific Library)?" % err)
            raise #errors.InvalidModelError(err)
        # create parrot neurons if necessary
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            self.all_cells_source = numpy.array(self.all_cells)  # we put the parrots into all_cells, since this will
            self.all_cells = nest.Create("parrot_neuron", self.size)     # be used for connections and recording. all_cells_source
            nest.Connect(self.all_cells_source, self.all_cells)  # should be used for setting parameters
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            for gid, source in zip(self.all_cells, self.all_cells_source):
                gid.source = source

    def _set_initial_value_array(self, variable, value):
        variable = STATE_VARIABLE_MAP.get(variable, variable)
        value = value.evaluate(simplify=True)
        try:
            nest.SetStatus(self.local_cells.tolist(), variable, value)
        except nest.NESTError, e:
            if "Unused dictionary items" in e.message:
                logger.warning("NEST does not allow setting an initial value for %s" % variable)
                # should perhaps check whether value-to-be-set is the same as current value,
                # and raise an Exception if not, rather than just emit a warning.
            else:
                raise
