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

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import descriptions, errors, models
from pyNN.parameters import ParameterSpace
import numpy
from pyNN.core import is_listlike
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
    def translated_parameters(self):
        return self.translate(self.parameter_space)

    def translate(self, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        _parameters = deepcopy(parameters)
        cls = self.__class__
        if parameters.schema != cls.get_schema():
            raise Exception("Schemas do not match: %s != %s" % (parameters.schema, cls.get_schema())) # should replace this with a PyNN-specific exception type
        native_parameters = {}
        #for name in parameters.schema:
        for name in parameters.keys():
            D = self.translations[name]
            pname = D['translated_name']
            try:
                pval = eval(D['forward_transform'], globals(), _parameters)
            except NameError, errmsg:
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
                try:
                    standard_parameters[name] = eval(D['reverse_transform'], {}, native_parameters)
                except NameError, errmsg:
                    raise NameError("Problem translating '%s' in %s. Transform: '%s'. Parameters: %s. %s" \
                                    % (name, cls.__name__, D['reverse_transform'], native_parameters, errmsg))
        return ParameterSpace(standard_parameters, schema=cls.get_schema(), shape=native_parameters.shape)

    def simple_parameters(self):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return [name for name in self.translations if self.translations[name]['forward_transform'] == name]

    def scaled_parameters(self):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        return [name for name in self.translations if "float" in self.translations[name]['forward_transform']]

    def computed_parameters(self):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return [name for name in self.translations if name not in self.simple_parameters() + self.scaled_parameters()]

    def get_translated_names(self, *names):
        if names:
            translations = (self.translations[name] for name in names)
        else:  # return all names
            translations = self.translations.itervalues()
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

class StandardSynapseType(StandardModelType, models.BaseSynapseType):
    pass


class STDPWeightDependence(StandardModelType):
    """Base class for models of STDP weight dependence."""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)


class STDPTimingDependence(StandardModelType):
    """Base class for models of STDP timing dependence (triplets, etc)"""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)
