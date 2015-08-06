"""
Base classes for cell and synapse models, whether "standard" (cross-simulator)
or "native" (restricted to an individual simulator).

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import descriptions
from pyNN.parameters import ParameterSpace


class BaseModelType(object):
    """Base class for standard and native cell and synapse model classes."""
    default_parameters = {}
    default_initial_values = {}
    parameter_checks = {}

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
        """Does this model have a parameter with the given name?"""
        return name in cls.default_parameters

    @classmethod
    def get_parameter_names(cls):
        """Return the names of the parameters of this model."""
        return cls.default_parameters.keys()

    def get_schema(self):
        """
        Returns the model schema: i.e. a mapping of parameter names to allowed
        parameter types.
        """
        return dict((name, type(value))
                    for name, value in self.default_parameters.items())

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
    
    def can_record(self, variable):
        return (variable in self.recordable)


class BaseCurrentSource(BaseModelType):
    """Base class for current source model classes."""
    pass


class BaseSynapseType(BaseModelType):
    """Base class for synapse model classes."""
    
    connection_type = None # override to specify a non-standard connection type (i.e. GapJunctions)
    has_presynaptic_components = False # override for synapses that include an active presynaptic components 

    def __init__(self, **parameters):
        """
        `parameters` should be a mapping object, e.g. a dict
        """
        all_parameters = self.default_parameters.copy()
        if parameters:
            all_parameters.update(**parameters)
        try:
            if all_parameters['delay'] is None:
                all_parameters['delay'] = self._get_minimum_delay()
            if all_parameters['weight'] is None:
                all_parameters['weight'] = 0.
        except KeyError as e:
            if e.args[0] != 'delay':  # ElectricalSynapses don't have delays
                raise e
        self.parameter_space = ParameterSpace(all_parameters,
                                              self.get_schema(),
                                              shape=None)
