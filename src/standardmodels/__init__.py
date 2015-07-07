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
    STDPWeightDependence
    STDPTimingDependence

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import errors, models
from pyNN.parameters import ParameterSpace
import numpy
from pyNN.core import is_listlike, itervalues
from copy import deepcopy

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
    extra_parameters = {}

    @property
    def native_parameters(self):
        """
        A :class:`ParameterSpace` containing parameter names and values
        translated from the standard PyNN names and units to simulator-specific
        ("native") names and units.
        """
        return self.translate(self.parameter_space)

    def translate(self, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        _parameters = deepcopy(parameters)
        cls = self.__class__
        if parameters.schema != self.get_schema():
            raise Exception("Schemas do not match: %s != %s" % (parameters.schema, self.get_schema())) # should replace this with a PyNN-specific exception type
        native_parameters = {}
        #for name in parameters.schema:
        for name in parameters.keys():
            D = self.translations[name]
            pname = D['translated_name']
            if callable(D['forward_transform']):
                pval = D['forward_transform'](**_parameters)
            else:
                try:
                    pval = eval(D['forward_transform'], globals(), _parameters)
                except NameError as errmsg:
                    raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s" \
                                    % (pname, cls.__name__, D['forward_transform'], parameters, errmsg))
                except ZeroDivisionError:
                    raise
                    #pval = 1e30 # this is about the highest value hoc can deal with
            native_parameters[pname] = pval
        return ParameterSpace(native_parameters, schema=None, shape=parameters.shape)

    def reverse_translate(self, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        cls = self.__class__
        standard_parameters = {}
        for name,D  in self.translations.items():
            tname = D['translated_name']
            if tname in native_parameters.keys():
                if callable(D['reverse_transform']):
                    standard_parameters[name] = D['reverse_transform'](**native_parameters)
                else:
                    try:
                        standard_parameters[name] = eval(D['reverse_transform'], {}, native_parameters)
                    except NameError as errmsg:
                        raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s" \
                                        % (name, cls.__name__, D['reverse_transform'], native_parameters, errmsg))
        return ParameterSpace(standard_parameters, schema=self.get_schema(), shape=native_parameters.shape)

    def simple_parameters(self):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return [name for name in self.translations if self.translations[name]['forward_transform'] == name]

    def scaled_parameters(self):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        def scaling(trans):
            return (not callable(trans)) and ("float" in trans)
        return [name for name in self.translations if scaling(self.translations[name]['forward_transform'])]

    def computed_parameters(self):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return [name for name in self.translations if name not in self.simple_parameters() + self.scaled_parameters()]

    def get_native_names(self, *names):
        """
        Return a list of native parameter names for a given model.
        """
        if names:
            translations = (self.translations[name] for name in names)
        else:  # return all names
            translations = itervalues(self.translations)
        return [D['translated_name'] for D in translations]


class StandardCellType(StandardModelType, models.BaseCellType):
    """Base class for standardized cell model classes."""
    recordable    = ['spikes', 'v', 'gsyn']
    receptor_types = ('excitatory', 'inhibitory')
    always_local  = False # override for NEST spike sources


class StandardCurrentSource(StandardModelType, models.BaseCurrentSource):
    """Base class for standardized current source model classes."""

    def inject_into(self, cells):
        """
        Inject the current from this source into the supplied group of cells.

        `cells` may be a :class:`Population`, :class:`PopulationView`,
        :class:`Assembly` or a list of :class:`ID` objects.
        """
        raise NotImplementedError("Should be redefined in the local simulator electrodes")

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
        else:
            parameters = ParameterSpace(parameters, self.get_schema(), (1,))
        parameters = self.translate(parameters)
        self.set_native_parameters(parameters)

    def get_parameters(self):
        """Return a dict of all current source parameters."""
        parameters = self.get_native_parameters()
        parameters = self.reverse_translate(parameters)
        return parameters

    def set_native_parameters(self, parameters):
        raise NotImplementedError

    def get_native_parameters(self):
        raise NotImplementedError


class ModelNotAvailable(object):
    """Not available for this simulator."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The %s model is not available for this simulator." % self.__class__.__name__)


# ==============================================================================
#   Synapse Dynamics classes
# ==============================================================================


def check_weights(weights, projection):
    # if projection.post is an Assembly, some components might have cond-synapses, others curr, so need a more sophisticated check here
    synapse_sign = projection.receptor_type
    is_conductance = projection.post.conductance_based
    if isinstance(weights, numpy.ndarray):
        all_negative = (weights <= 0).all()
        all_positive = (weights >= 0).all()
        if not (all_negative or all_positive):
            raise errors.ConnectionError("Weights must be either all positive or all negative")
    elif numpy.isreal(weights):
        all_positive = weights >= 0
        all_negative = weights < 0
    else:
        raise errors.ConnectionError("Weights must be a number or an array of numbers.")
    if is_conductance or synapse_sign == 'excitatory':
        if not all_positive:
            raise errors.ConnectionError("Weights must be positive for conductance-based and/or excitatory synapses")
    elif is_conductance == False and synapse_sign == 'inhibitory':
        if not all_negative:
            raise errors.ConnectionError("Weights must be negative for current-based, inhibitory synapses")
    else:  # This should never happen.
        raise Exception("Can't check weight, conductance status unknown.")


def check_delays(delays, projection):
    min_delay = projection._simulator.state.min_delay
    max_delay = projection._simulator.state.max_delay
    if isinstance(delays, numpy.ndarray):
        below_max = (delays <= max_delay).all()
        above_min = (delays >= min_delay).all()
        in_range = below_max and above_min
    elif numpy.isreal(delays):
        in_range = min_delay <= delays <= max_delay
    else:
        raise errors.ConnectionError("Delays must be a number or an array of numbers.")
    if not in_range:
        raise errors.ConnectionError("Delay (%s) is out of range [%s, %s]" % (delays, min_delay, max_delay))


class StandardSynapseType(StandardModelType, models.BaseSynapseType):
    parameter_checks = {
        'weight': check_weights,
        'delay': check_delays
    }

    def get_schema(self):
        """
        Returns the model schema: i.e. a mapping of parameter names to allowed
        parameter types.
        """
        base_schema = dict((name, type(value))
                           for name, value in self.default_parameters.items())
        base_schema['delay'] = float  # delay has default value None, meaning "use the minimum delay", so we have to correct the auto-generated schema
        return base_schema


class STDPWeightDependence(StandardModelType):
    """Base class for models of STDP weight dependence."""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)


class STDPTimingDependence(StandardModelType):
    """Base class for models of STDP timing dependence (triplets, etc)"""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)
