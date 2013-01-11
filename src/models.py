"""
Base classes for cell and synapse models, whether "standard" (cross-simulator)
or "native" (restricted to an individual simulator).

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import descriptions
from pyNN.parameters import ParameterSpace


class BaseModelType(object):
    """Base class for standard and native cell and synapse model classes."""
    default_parameters = {}
    default_initial_values = {}

    def __init__(self, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        self.parameter_space = ParameterSpace(self.default_parameters,
                                              self.get_schema(),
                                              shape=None)
        if parameters:
            self.parameter_space.update(**parameters)

    def __repr__(self):
        return "%s(<parameters>)" % self.__class__.__name__ # should really include the parameters explicitly, to be unambiguous

    @classmethod
    def has_parameter(cls, name):
        return name in cls.default_parameters

    @classmethod
    def get_parameter_names(cls):
        return cls.default_parameters.keys()

    @classmethod
    def get_schema(cls):
        """
        Returns the model schema: i.e. a mapping of parameter names to allowed
        parameter types.
        """
        return dict((name, type(value))
                    for name, value in cls.default_parameters.items())

    def describe(self, template='modeltype_default.txt', engine='default'):
        """
        Returns a human-readable description of the cell or synapse type.

        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {
            "name": self.__class__.__name__,
            "default_parameters": self.default_parameters,
            "default_initial_values": self.default_initial_values,
            "parameters": self.parameter_space._parameters, # should add a describe() method to ParameterSpace
        }
        return descriptions.render(engine, template, context)


class BaseCellType(BaseModelType):
    """Base class for cell model classes."""
    recordable = []
    receptor_types = []
    conductance_based = True # override for cells with current-based synapses
    injectable = True # override for spike sources


class BaseCurrentSource(BaseModelType):
    """Base class for current source model classes."""
    pass


class BaseSynapseType(BaseModelType):
    """Base class for synapse model classes."""

    def __init__(self, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        all_parameters = self.default_parameters.copy()
        if parameters:
            all_parameters.update(**parameters)
        if all_parameters['delay'] is None:
            all_parameters['delay'] = self._get_minimum_delay()
        self.parameter_space = ParameterSpace(all_parameters,
                                              self.get_schema(),
                                              shape=None)
