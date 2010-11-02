# encoding: utf-8
"""
Machinery for implementation of "standard models", i.e. neuron and synapse models
that are available in multiple simulators:

Functions:
    build_translations()
    
Classes:
    StandardModelType
    StandardCellType
    ModelNotAvailable
    SynapseDynamics
    ShortTermPlasticityMechanism
    STDPMechanism
    STDPWeightDependence
    STDPTimingDependence
    
"""

import copy
import descriptions
import numpy
from core import is_listlike
import errors
from string import Template

# ==============================================================================
#   Standard cells
# ==============================================================================

def build_translations(*translation_list):
    """
    Build a translation dictionary from a list of translations/transformations.
    """
    translations = {}
    for item in translation_list:
        assert 2 <= len(item) <= 4, "Translation tuples must have between 2 and 4 items. Actual content: %s" % str(item)
        pynn_name = item[0]
        sim_name = item[1]
        if len(item) == 2: # no transformation
            f = pynn_name
            g = sim_name
        elif len(item) == 3: # simple multiplicative factor
            scale_factor = item[2]
            f = "float(%g)*%s" % (scale_factor, pynn_name)
            g = "%s/float(%g)" % (sim_name, scale_factor)
        elif len(item) == 4: # more complex transformation
            f = item[2]
            g = item[3]
        translations[pynn_name] = {'translated_name': sim_name,
                                   'forward_transform': f,
                                   'reverse_transform': g}
    return translations


class StandardModelType(object):
    """Base class for standardized cell model and synapse model classes."""
    
    translations = {}
    default_parameters = {}
    default_initial_values = {}
    
    def __init__(self, parameters):
        self.parameters = self.__class__.checkParameters(parameters, with_defaults=True)
        self.parameters = self.__class__.translate(self.parameters)
    
    @classmethod
    def has_parameter(cls, name):
        return name in cls.default_parameters
    
    @classmethod
    def get_parameter_names(cls):
        return cls.default_parameters.keys()
    
    @classmethod
    def checkParameters(cls, supplied_parameters, with_defaults=False):
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
                if default_parameters.has_key(k):
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
    
    @classmethod
    def translate(cls, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        parameters = cls.checkParameters(parameters, with_defaults=False)
        native_parameters = {}
        for name in parameters:
            D = cls.translations[name]
            pname = D['translated_name']
            if is_listlike(cls.default_parameters[name]):
                parameters[name] = numpy.array(parameters[name])
            try:
                pval = eval(D['forward_transform'], globals(), parameters)
            except NameError, errmsg:
                raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s" \
                                % (pname, cls.__name__, D['forward_transform'], parameters, errmsg))
            except ZeroDivisionError:
                raise
                #pval = 1e30 # this is about the highest value hoc can deal with
            native_parameters[pname] = pval
        return native_parameters
    
    @classmethod
    def reverse_translate(cls, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        standard_parameters = {}
        for name,D  in cls.translations.items():
            if is_listlike(cls.default_parameters[name]):
                tname = D['translated_name']
                native_parameters[tname] = numpy.array(native_parameters[tname])
            try:
                standard_parameters[name] = eval(D['reverse_transform'], {}, native_parameters)
            except NameError, errmsg:
                raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s" \
                                % (name, cls.__name__, D['reverse_transform'], native_parameters, errmsg))
        return standard_parameters

    @classmethod
    def simple_parameters(cls):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return [name for name in cls.translations if cls.translations[name]['forward_transform'] == name]

    @classmethod
    def scaled_parameters(cls):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        return [name for name in cls.translations if "float" in cls.translations[name]['forward_transform']]
    
    @classmethod
    def computed_parameters(cls):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return [name for name in cls.translations if name not in cls.simple_parameters()+cls.scaled_parameters()]
        
    def update_parameters(self, parameters):
        """
        update self.parameters with those in parameters 
        """
        self.parameters.update(self.translate(parameters))
    
    def describe(self, template='standardmodeltype_default.txt', engine='default'):
        """
        Returns a human-readable description of the cell type.
        
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


class StandardCellType(StandardModelType):
    """Base class for standardized cell model classes."""

    recordable = ['spikes', 'v', 'gsyn']
    synapse_types = ('excitatory', 'inhibitory')
    conductance_based = True # over-ride for cells with current-based synapses
    always_local = False # over-ride for NEST spike sources


class ModelNotAvailable(object):
    """Not available for this simulator."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The %s model is not available for this simulator." % self.__class__.__name__)
        
# ==============================================================================
#   Synapse Dynamics classes
# ==============================================================================

class SynapseDynamics(object):
    """
    For specifying synapse short-term (faciliation, depression) and long-term
    (STDP) plasticity. To be passed as the `synapse_dynamics` argument to
    `Projection.__init__()` or `connect()`.
    """
    
    def __init__(self, fast=None, slow=None):
        """
        Create a new specification for a dynamic synapse, combining a `fast`
        component (short-term facilitation/depression) and a `slow` component
        (long-term potentiation/depression).
        """
        if fast:
            assert isinstance(fast, ShortTermPlasticityMechanism)
        if slow:
            assert isinstance(slow, STDPMechanism)
            assert 0 <= slow.dendritic_delay_fraction <= 1.0
        self.fast = fast
        self.slow = slow
    
    def describe(self, template='synapsedynamics_default.txt', engine='default'):
        """
        Returns a human-readable description of the synapse dynamics.
        
        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).
        
        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {'fast': self.fast and self.fast.describe(template=None) or 'None',
                   'slow': self.slow and self.slow.describe(template=None) or 'None'}
        return descriptions.render(engine, template, context)


class ShortTermPlasticityMechanism(StandardModelType):
    """Abstract base class for models of short-term synaptic dynamics."""
    
    def __init__(self):
        raise NotImplementedError


class STDPMechanism(object):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        """
        Create a new specification for an STDP mechanism, by combining a
        weight-dependence, a timing-dependence, and, optionally, a voltage-
        dependence.
        
        For point neurons, the synaptic delay `d` can be interpreted either as
        occurring purely in the pre-synaptic axon + synaptic cleft, in which
        case the synaptic plasticity mechanism 'sees' the post-synaptic spike
        immediately and the pre-synaptic spike after a delay `d`
        (`dendritic_delay_fraction = 0`) or as occurring purely in the post-
        synaptic dendrite, in which case the pre-synaptic spike is seen
        immediately, and the post-synaptic spike after a delay `d`
        (`dendritic_delay_fraction = 1`), or as having both pre- and post-
        synaptic components (`dendritic_delay_fraction` between 0 and 1).
        
        In a future version of the API, we will allow the different
        components of the synaptic delay to be specified separately in
        milliseconds.
        """
        if timing_dependence:
            assert isinstance(timing_dependence, STDPTimingDependence)
        if weight_dependence:
            assert isinstance(weight_dependence, STDPWeightDependence)
        assert isinstance(dendritic_delay_fraction, (int, long, float))
        assert 0 <= dendritic_delay_fraction <= 1
        self.timing_dependence = timing_dependence
        self.weight_dependence = weight_dependence
        self.voltage_dependence = voltage_dependence
        self.dendritic_delay_fraction = dendritic_delay_fraction
    
    @property
    def possible_models(self):
        td = self.timing_dependence
        wd = self.weight_dependence
        pm = td.possible_models.intersection(wd.possible_models)
        if len(pm) == 1 :
            return list(pm)[0]
        elif len(pm) == 0 :
            raise NoModelAvailableError("No available plasticity models")
        elif len(pm) > 1 :
            # we pass the set of models back to the simulator-specific module for it to deal with
            return pm
    
    @property
    def all_parameters(self):
        parameters = self.timing_dependence.parameters.copy()
        parameters.update(self.weight_dependence.parameters)
        return parameters
    
    def describe(self, template='stdpmechanism_default.txt', engine='default'):
        """
        Returns a human-readable description of the STDP mechanism.
        
        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).
        
        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {'weight_dependence': self.weight_dependence.describe(template=None),
                   'timing_dependence': self.timing_dependence.describe(template=None),
                   'voltage_dependence': self.voltage_dependence and self.voltage_dependence.describe(template=None) or None,
                   'dendritic_delay_fraction': self.dendritic_delay_fraction}
        return descriptions.render(engine, template, context)


class STDPWeightDependence(StandardModelType):
    """Abstract base class for models of STDP weight dependence."""
    
    def __init__(self):
        raise NotImplementedError


class STDPTimingDependence(StandardModelType):
    """Abstract base class for models of STDP timing dependence (triplets, etc)"""
    
    def __init__(self):
        raise NotImplementedError
