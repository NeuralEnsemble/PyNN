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
   
:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import descriptions, errors, models
import numpy
from pyNN.core import is_listlike

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


class StandardModelType(models.BaseModelType):
    """Base class for standardized cell model and synapse model classes."""
    
    translations = {}
    
    def __init__(self, parameters):
        models.BaseModelType.__init__(self, parameters)
        assert set(self.translations.keys()) == set(self.default_parameters.keys()), \
               "%s != %s" % (self.translations.keys(), self.default_parameters.keys())
        self.parameters = self.__class__.translate(self.parameters)
    
    @classmethod
    def translate(cls, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        parameters = cls.check_parameters(parameters, with_defaults=False)
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


class StandardCellType(StandardModelType, models.BaseCellType):
    """Base class for standardized cell model classes."""
    recordable    = ['spikes', 'v', 'gsyn']
    synapse_types = ('excitatory', 'inhibitory')
    always_local  = False # override for NEST spike sources


class StandardCurrentSource(StandardModelType, models.BaseCurrentSource):
    """Base class for standardized current source model classes."""             
    
    def inject_into(self, cells):
        raise Exception("Should be redefined in the local simulator electrodes")

    def __getattr__(self, name):
        try:
            val = self.__getattribute__(name)
        except AttributeError:
            try:
                val = self.get_parameters()[name]
            except KeyError:
                raise errors.NonExistentParameterError(name,
                                                       self.__class__.__name__,
                                                       self.get_parameter_names())
        return val

    def __setattr__(self, name, value):
        if self.has_parameter(name):
            self.set_parameters(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def set_parameters(self, **parameters):
        """
        Set current source parameters, given as a sequence of parameter=value arguments.
        """
        # if some of the parameters are computed from the values of other
        # parameters, need to get and translate all parameters
        computed_parameters = self.computed_parameters()
        have_computed_parameters = numpy.any([p_name in computed_parameters
                                              for p_name in parameters])
        if have_computed_parameters:
            all_parameters = self.get_parameters()
            all_parameters.update(parameters)
            parameters = all_parameters
            parameters = self.translate(parameters)
        self.set_native_parameters(parameters)

    def get_parameters(self):
        """Return a dict of all current source parameters."""
        parameters = self.get_native_parameters()
        parameters = self.reverse_translate(parameters)
        return parameters

    def set_native_parameters(self, parameters):
        pass

    def get_native_parameters(self):    
        pass

class ModelNotAvailable(object):
    """Not available for this simulator."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The %s model is not available for this simulator." % self.__class__.__name__)
        
# ==============================================================================
#   Synapse Dynamics classes
# ==============================================================================

class SynapseDynamics(models.BaseSynapseDynamics):
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
        context = {'fast': self.fast and self.fast.describe(template=None) or None,
                   'slow': self.slow and self.slow.describe(template=None) or None}
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
            raise errors.NoModelAvailableError("No available plasticity models")
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
