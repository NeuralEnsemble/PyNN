"""
Base classes for cell and synapse models, whether "standard" (cross-simulator)
or "native" (restricted to an individual simulator).

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import copy
from pyNN import errors, descriptions

class BaseModelType(object):
    """Base class for standard and native cell and synapse model classes."""
    # does not really belong in this module. Some reorganisation required.
    default_parameters = {}
    default_initial_values = {}

    def __init__(self, parameters):
        self.parameters = self.__class__.check_parameters(parameters, with_defaults=True)

    @classmethod
    def has_parameter(cls, name):
        return name in cls.default_parameters
    
    @classmethod
    def get_parameter_names(cls):
        return cls.default_parameters.keys()
    
    @classmethod
    def check_parameters(cls, supplied_parameters, with_defaults=False):
        """
        Returns a parameter dictionary, checking that each
        supplied_parameter is in the default_parameters and
        converts to the type of the latter.

        If with_defaults==True, parameters not in
        supplied_parameters are in the returned dictionary
        as in default_parameters.

        """
        default_parameters = cls.default_parameters
        if with_defaults:
            parameters = copy.copy(default_parameters)
        else:
            parameters = {}
        if supplied_parameters:
            for k in supplied_parameters.keys():
                if k in default_parameters.keys():
                    err_msg = "For %s in %s, expected %s, got %s (%s)" % \
                              (k, cls.__name__, type(default_parameters[k]),
                               type(supplied_parameters[k]), supplied_parameters[k])
                    # same type
                    if type(supplied_parameters[k]) == type(default_parameters[k]): 
                        parameters[k] = supplied_parameters[k]
                    # float and something that can be converted to a float
                    elif isinstance(default_parameters[k], float): 
                        try:
                            parameters[k] = float(supplied_parameters[k]) 
                        except (ValueError, TypeError):
                            raise errors.InvalidParameterValueError(err_msg)
                    # list and something that can be transformed to a list
                    elif isinstance(default_parameters[k], list):
                        try:
                            parameters[k] = list(supplied_parameters[k])
                        except TypeError:
                            raise errors.InvalidParameterValueError(err_msg)
                    else:
                        raise errors.InvalidParameterValueError(err_msg)
                else:
                    raise errors.NonExistentParameterError(k, cls, cls.default_parameters.keys())
        return parameters

    def describe(self, template='modeltype_default.txt', engine='default'):
        """
        Returns a human-readable description of the cll or synapse type.
        
        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).
        
        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {
            "name": self.__class__.__name__,
            "parameters": self.parameters,
        }
        return descriptions.render(engine, template, context)


class BaseCellType(BaseModelType):
    """Base class for cell model classes."""
    recordable = []
    synapse_types = []
    conductance_based = True # override for cells with current-based synapses
    injectable = True # override for spike sources

class BaseCurrentSource(BaseModelType):
    """Base class for current source model classes."""

class BaseSynapseDynamics(object):
    pass
