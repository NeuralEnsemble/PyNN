# -*- coding: utf-8 -*-
"""
Support cell types defined in NESTML (https://nestml.readthedocs.org/).

Requires NESTML to be installed.

:copyright: Copyright 2006-2024 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import pynestml
import pynestml.frontend
import pynestml.frontend.pynestml_frontend
from pynestml.utils.model_parser import ModelParser
from pynestml.codegeneration.python_standalone_target_tools import PythonStandaloneTargetTools
from pyNN.nest.cells import NativeCellType


logger = logging.getLogger("PyNN")


class NESTMLCellType(NativeCellType):

    def __init__(self, parameters):
        NativeCellType.__init__(self, parameters)


def nestml_celltype_from_model(nestml_file_name: str):
    """
    Return a new NativeCellType subclass from a NESTML model.
    """

    dct = {'nestml_file_name': nestml_file_name}
    return _nest_build_nestml_celltype((NESTMLCellType,), dct)


class _nest_build_nestml_celltype(type):
    """
    Metaclass for building NESTMLCellType subclasses
    """
    def __new__(cls, bases, dct):
        import nest
        import pynestml

        nestml_file_name = dct['nestml_file_name']

        pynestml.frontend.pynestml_frontend.generate_target(input_path=nestml_file_name,
                                                            target_platform="NEST",
                                                            suffix="_nestml",
                                                            logging_level="WARNING")

        ast_compilation_unit = ModelParser.parse_file(nestml_file_name)
        if ast_compilation_unit is None or len(ast_compilation_unit.get_model_list()) == 0:
            raise("Error(s) occurred during code generation; please check error messages")

        model: ASTModel = ast_compilation_unit.get_model_list()[0]
        model_name = model.get_name()

        dct["default_parameters"], dct["default_initial_values"] = PythonStandaloneTargetTools.get_neuron_parameters_and_state(nestml_file_name)
        dct["synapse_types"] = [port.name for port in model.get_spike_input_ports()]
        dct["standard_receptor_type"] = ()
        dct["injectable"] = bool(model.get_continuous_input_ports())  # assume that in case there is a continuous-time input port, it corresponds with a current injection port
        dct["conductance_based"] = False    # this is only used for checking sign of the incoming weights -- assume always false to skip the check
        dct["model_name"] = model_name
        dct["nest_model"] = model_name

        # Recording from bindings:
        dct["recordable"] = dct["default_initial_values"].keys()
        # XXX TODO: add recordable inlines

        dct["weight_variables"] = []   # XXX: none for neuron models, no?

        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (model_name, bases, dct))

        return type.__new__(cls, model_name, bases, dct)
