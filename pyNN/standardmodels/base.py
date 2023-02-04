"""
Base classes for standard models

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import warnings
from copy import deepcopy

import numpy as np
import neo
import quantities as pq

from .. import errors, models
from ..parameters import ParameterSpace


excitatory_receptor_types = ["excitatory", "AMPA", "NMDA"]
inhibitory_receptor_types = ["inhibitory", "GABA", "GABAA", "GABAB"]

# ==============================================================================
#   Standard cells
# ==============================================================================


def build_scaling_functions(pynn_name, sim_name, scale_factor):
    def f(**p):
        return p[pynn_name] * scale_factor

    def g(**p):
        return p[sim_name] / scale_factor
    return f, g


def build_translations(*translation_list):
    """
    Build a translation dictionary from a list of translations/transformations.
    """
    translations = {}
    for item in translation_list:
        err_msg = f"Translation tuples must have between 2 and 4 items. Actual content: {item}"
        assert 2 <= len(item) <= 4, err_msg
        pynn_name = item[0]
        sim_name = item[1]
        if len(item) == 2:  # no transformation
            f = pynn_name
            g = sim_name
            type_ = "simple"
        elif len(item) == 3:  # simple multiplicative factor
            scale_factor = item[2]
            f, g = build_scaling_functions(pynn_name, sim_name, scale_factor)
            type_ = "scaled"
        elif len(item) == 4:  # more complex transformation
            f = item[2]
            g = item[3]
            type_ = "computed"
        translations[pynn_name] = {'translated_name': sim_name,
                                   'forward_transform': f,
                                   'reverse_transform': g,
                                   'type': type_}
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

    def translate(self, parameters, copy=True):
        """Translate standardized model parameters to simulator-specific parameters."""
        if copy:
            _parameters = deepcopy(parameters)
        else:
            _parameters = parameters
        cls = self.__class__
        if parameters.schema != self.get_schema():
            # should replace this with a PyNN-specific exception type
            raise Exception(f"Schemas do not match: {parameters.schema} != {self.get_schema()}")
        native_parameters = {}
        for name in parameters.keys():
            D = self.translations[name]
            pname = D['translated_name']
            if callable(D['forward_transform']):
                pval = D['forward_transform'](**_parameters)
            else:
                try:
                    pval = eval(D['forward_transform'], globals(), _parameters)
                except NameError as err:
                    raise NameError(
                        f"Problem translating '{pname}' in {cls.__name__}. "
                        f"Transform: '{D['forward_transform']}'. Parameters: {parameters}. {err}"
                    )
                except ZeroDivisionError:
                    raise
            native_parameters[pname] = pval
        return ParameterSpace(native_parameters, schema=None, shape=parameters.shape)

    def reverse_translate(self, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        cls = self.__class__
        standard_parameters = {}
        for name, D in self.translations.items():
            tname = D['translated_name']
            if tname in native_parameters.keys():
                if callable(D['reverse_transform']):
                    standard_parameters[name] = D['reverse_transform'](**native_parameters)
                else:
                    try:
                        standard_parameters[name] = eval(
                            D['reverse_transform'], {}, native_parameters)
                    except NameError as err:
                        raise NameError(
                            f"Problem translating '{name}' in {cls.__name__}. "
                            f"Transform: '{D['reverse_transform']}'. "
                            f"Parameters: {native_parameters}. {err}"
                        )
        return ParameterSpace(standard_parameters,
                              schema=self.get_schema(),
                              shape=native_parameters.shape)

    def simple_parameters(self):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return [name for name in self.translations
                if self.translations[name]['type'] == "simple"]

    def scaled_parameters(self):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        return [name for name in self.translations
                if self.translations[name]['type'] == "scaled"]

    def computed_parameters(self):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return [name for name in self.translations
                if self.translations[name]['type'] == "computed"]

    def computed_parameters_include(self, parameter_names):
        return any(name in self.computed_parameters() for name in parameter_names)

    def get_native_names(self, *names):
        """
        Return a list of native parameter names for a given model.
        """
        if names:
            translations = (self.translations[name] for name in names)
        else:  # return all names
            translations = self.translations.values()
        return [D['translated_name'] for D in translations]


class StandardCellType(StandardModelType, models.BaseCellType):
    """Base class for standardized cell model classes."""
    recordable = ['spikes', 'v', 'gsyn']
    receptor_types = ('excitatory', 'inhibitory')
    always_local = False  # override for NEST spike sources


class StandardCellTypeComponent(StandardModelType, models.BaseCellTypeComponent):
    """docstring needed"""
    pass


class StandardPostSynapticResponse(StandardModelType, models.BasePostSynapticResponse):
    """docstring needed"""

    def set_parent(self, parent):
        """

        """
        self.parent = parent


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
        if name == "set":
            err_msg = "For current sources, set values using the parameter name directly, " \
                      "e.g. source.amplitude = 0.5, or use 'set_parameters()' " \
                      "e.g. source.set_parameters(amplitude=0.5)"
            raise AttributeError(err_msg)

        try:
            val = self.get_parameters()[name]
        except KeyError:
            try:
                val = self.__getattribute__(name)
            except AttributeError:
                raise errors.NonExistentParameterError(name,
                                                       self.__class__.__name__,
                                                       self.get_parameter_names())
        return val

    def __setattr__(self, name, value):
        if self.has_parameter(name):
            self.set_parameters(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def set_parameters(self, copy=True, **parameters):
        """
        Set current source parameters, given as a sequence of parameter=value arguments.
        """
        # if some of the parameters are computed from the values of other
        # parameters, need to get and translate all parameters
        if self.computed_parameters_include(parameters):
            all_parameters = self.get_parameters()
            all_parameters.update(parameters)
            parameters = all_parameters
        else:
            parameters = ParameterSpace(parameters, self.get_schema(), (1,))
        parameters = self.translate(parameters, copy=copy)
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

    def _round_timestamp(self, value, resolution):
        # todo: consider using decimals module,
        # since rounding of floating point numbers is so horrible
        return np.rint(value / resolution) * resolution

    def get_data(self):
        """Return the recorded current as a Neo signal object"""
        t_arr, i_arr = self._get_data()
        intervals = np.diff(t_arr)
        if intervals.size > 0 and intervals.max() - intervals.min() < 1e-9:
            signal = neo.AnalogSignal(i_arr, units="nA", t_start=t_arr[0] * pq.ms,
                                      sampling_period=intervals[0] * pq.ms)
        else:
            signal = neo.IrregularlySampledSignal(t_arr, i_arr, units="nA", time_units="ms")
        return signal


class ModelNotAvailable(object):
    """Not available for this simulator."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"The {self.__class__.__name__} model is not available for this simulator.")


# ==============================================================================
#   Synapse Dynamics classes
# ==============================================================================


def check_weights(weights, projection):
    # if projection.post is an Assembly, some components might have cond-synapses, others curr,
    # so need a more sophisticated check here. For now, skipping check and emitting a warning
    if (
        hasattr(projection.post, "_homogeneous_synapses")
        and not projection.post._homogeneous_synapses     # noqa: W503
    ):
        warnings.warn("Not checking weights due to due mixture of synapse types")
    if isinstance(weights, np.ndarray):
        all_negative = (weights <= 0).all()
        all_positive = (weights >= 0).all()
        if not (all_negative or all_positive):
            raise errors.ConnectionError("Weights must be either all positive or all negative")
    elif np.isreal(weights):
        all_positive = weights >= 0
        all_negative = weights <= 0
    else:
        raise errors.ConnectionError("Weights must be a number or an array of numbers.")
    if projection.post.conductance_based or projection.receptor_type in excitatory_receptor_types:
        if not all_positive:
            raise errors.ConnectionError(
                "Weights must be positive for conductance-based and/or excitatory synapses"
            )
    elif (
        projection.post.conductance_based is False
        and projection.receptor_type in inhibitory_receptor_types  # noqa: W503
    ):
        if not all_negative:
            raise errors.ConnectionError(
                "Weights must be negative for current-based, inhibitory synapses"
            )
    else:
        # This can happen for multi-synapse models
        # if the receptor_type is not one of the commonly used ones
        warnings.warn("Can't check weight, conductance status unknown.")


def check_delays(delays, projection):
    min_delay = projection._simulator.state.min_delay
    max_delay = projection._simulator.state.max_delay
    if isinstance(delays, np.ndarray):
        below_max = (delays <= max_delay).all()
        above_min = (delays >= min_delay).all()
        in_range = below_max and above_min
    elif np.isreal(delays):
        in_range = min_delay <= delays <= max_delay
    else:
        raise errors.ConnectionError("Delays must be a number or an array of numbers.")
    if not in_range:
        raise errors.ConnectionError(
            f"Delay ({delays}) is out of range [{min_delay}, {max_delay}]")


class StandardSynapseType(StandardModelType, models.BaseSynapseType):
    parameter_checks = {
        'weight': check_weights,
        # 'delay': check_delays  # this needs to be revisited in the context of min_delay = "auto"
    }

    def get_schema(self):
        """
        Returns the model schema: i.e. a mapping of parameter names to allowed
        parameter types.
        """
        base_schema = dict((name, type(value))
                           for name, value in self.default_parameters.items())
        base_schema['delay'] = float
        # delay has default value None, meaning "use the minimum delay",
        # so we have to correct the auto-generated schema
        return base_schema


class STDPWeightDependence(StandardModelType):
    """Base class for models of STDP weight dependence."""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)


class STDPTimingDependence(StandardModelType):
    """Base class for models of STDP timing dependence (triplets, etc)"""

    def __init__(self, **parameters):
        StandardModelType.__init__(self, **parameters)
