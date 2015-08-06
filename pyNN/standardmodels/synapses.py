# encoding: utf-8
"""
Definition of default parameters (and hence, standard parameter names) for
standard dynamic synapse models.

Classes for specifying short-term plasticity (facilitation/depression):
    TsodyksMarkramSynapse

Classes for defining STDP rules:
    AdditiveWeightDependence
    MultiplicativeWeightDependence
    AdditivePotentiationMultiplicativeDepression
    GutigWeightDependence
    SpikePairRule

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    long
except NameError:
    long = int
from pyNN import descriptions
from pyNN.standardmodels import StandardSynapseType, STDPWeightDependence, STDPTimingDependence
from pyNN.parameters import ParameterSpace


class StaticSynapse(StandardSynapseType):
    """
    Synaptic connection with fixed weight and delay.
    """
    default_parameters = {
        'weight': 0.0,
        'delay': None
    }

class ElectricalSynapse(StandardSynapseType):
    """
    A bidirectional electrical synapse (gap junction) with fixed conductance
    """
       
    default_parameters = {
        'weight': 0.0 # the (bidirectional) conductance of the gap junction (uS) 
    }

class TsodyksMarkramSynapse(StandardSynapseType):
    """
    Synapse exhibiting facilitation and depression, implemented using the model
    of Tsodyks, Markram et al.:

    `Tsodyks, Uziel and Markram (2000)`_ Synchrony Generation in Recurrent Networks
    with Frequency-Dependent Synapses. Journal of Neuroscience 20:RC50

    Note that the time constant of the post-synaptic current is set in the
    neuron model, not here.

    Arguments:
        `U`:
            use parameter.
        `tau_rec`:
            depression time constant (ms).
        `tau_facil`:
            facilitation time constant (ms).
    
    .. _`Tsodyks, Uziel and Markram (2000)`: http://www.jneurosci.org/content/20/1/RC50.long
    """
    default_parameters = {
        'weight': 0.0,
        'delay': None,
        'U': 0.5,   # use parameter
        'tau_rec': 100.0, # depression time constant (ms)
        'tau_facil': 0.0,   # facilitation time constant (ms)
    }


class STDPMechanism(StandardSynapseType):
    """
    A specification for an STDP mechanism, combining a weight-dependence, a
    timing-dependence, and, optionally, a voltage-dependence of the synaptic
    change.

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

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        """
        Create a new specification for an STDP mechanism, by combining a
        weight-dependence, a timing-dependence, and, optionally, a voltage-
        dependence.
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
        self.weight = weight
        self.delay = delay or self._get_minimum_delay()
        self._build_translations()
        
    def _build_translations(self):
        self.translations = self.__class__.base_translations  # weight and delay
        for component in (self.timing_dependence, self.weight_dependence, self.voltage_dependence):
            if component:
                self.translations.update(component.translations)

    @property
    def model(self):
        return list(self.possible_models)[0]

    @property
    def possible_models(self):
        """
        A list of available synaptic plasticity models for the current
        configuration (weight dependence, timing dependence, ...) in the
        current simulator.
        """
        td = self.timing_dependence
        wd = self.weight_dependence
        pm = td.possible_models.intersection(wd.possible_models)
        if len(pm) == 0 :
            raise errors.NoModelAvailableError("No available plasticity models")
        else:
            # we pass the set of models back to the simulator-specific module for it to deal with
            return pm

    def get_parameter_names(self):
        assert self.voltage_dependence is None  # once we have some models with v-dep, need to update the following
        return ['weight', 'delay'] + self.timing_dependence.get_parameter_names() + self.weight_dependence.get_parameter_names()

    def get_schema(self):
        """
        Returns the model schema: i.e. a mapping of parameter names to allowed
        parameter types.
        """
        base_schema = {'weight': float, 'delay': float}
        for component in (self.timing_dependence, self.weight_dependence, self.voltage_dependence):
            if component:
                base_schema.update((name, type(value))
                                   for name, value in component.default_parameters.items())
        return base_schema

    @property
    def native_parameters(self):
        """
        A dictionary containing the combination of parameters from the different
        components of the STDP model.
        """
        timing_parameters = self.timing_dependence.native_parameters
        weight_parameters = self.weight_dependence.native_parameters
        parameters = self.translate(
                        ParameterSpace({'weight': self.weight,
                                        'delay': self.delay}, self.get_schema()))
        parameters.update(**timing_parameters)
        parameters.update(**weight_parameters)
        parameters.update(**self.timing_dependence.extra_parameters)
        parameters.update(**self.weight_dependence.extra_parameters)
        parameters.update(dendritic_delay_fraction=self.dendritic_delay_fraction)
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


class AdditiveWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change is independent of the current weight.
    If the new weight would be less than `w_min` it is set to `w_min`. If it
    would be greater than `w_max` it is set to `w_max`.

    Arguments:
        `w_min`:
            minimum synaptic weight, in the same units as the weight, i.e.
            µS or nA.
        `w_max`:
            maximum synaptic weight.
    """
    default_parameters = {
        'w_min':   0.0,
        'w_max':   1.0,
    }

    def __init__(self, w_min=0.0, w_max=1.0):
        parameters = dict(locals())
        parameters.pop('self')
        STDPWeightDependence.__init__(self, **parameters)


class MultiplicativeWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Δw ∝ w - w_min
    For potentiation, Δw ∝ w_max - w

    Arguments:
        `w_min`:
            minimum synaptic weight, in the same units as the weight, i.e.
            µS or nA.
        `w_max`:
            maximum synaptic weight.
    """
    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
    }

    def __init__(self, w_min=0.0, w_max=1.0):
        parameters = dict(locals())
        parameters.pop('self')
        STDPWeightDependence.__init__(self, **parameters)


class AdditivePotentiationMultiplicativeDepression(STDPWeightDependence):
    """
    The amplitude of the weight change depends on the current weight for
    depression (Δw ∝ w) and is fixed for potentiation.

    Arguments:
        `w_min`:
            minimum synaptic weight, in the same units as the weight, i.e.
            µS or nA.
        `w_max`:
            maximum synaptic weight.
    """

    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
    }

    def __init__(self, w_min=0.0,  w_max=1.0):
        parameters = dict(locals())
        parameters.pop('self')
        STDPWeightDependence.__init__(self, **parameters)


class GutigWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change depends on (w_max-w)^mu_plus for
    potentiation and (w-w_min)^mu_minus for depression.

    Arguments:
        `w_min`:
            minimum synaptic weight, in the same units as the weight, i.e.
            µS or nA.
        `w_max`:
            maximum synaptic weight.
        `mu_plus`:
            see above
        `mu_minus`:
            see above
    """

    default_parameters = {
        'w_min'   : 0.0,
        'w_max'   : 1.0,
        'mu_plus' : 0.5,
        'mu_minus': 0.5
    }

    def __init__(self, w_min=0.0,  w_max=1.0, mu_plus=0.5, mu_minus=0.5):
        """
        Create a new specification for the weight-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        STDPWeightDependence.__init__(self, **parameters)


# Not yet implemented for any module
#class PfisterSpikeTripletRule(STDPTimingDependence):
#    raise NotImplementedError


class SpikePairRule(STDPTimingDependence):
    """
    The amplitude of the weight change depends only on the relative timing of
    spike pairs, not triplets, etc. All possible spike pairs are taken into
    account (cf Song and Abbott).

    Arguments:
        `tau_plus`:
            time constant of the positive part of the STDP curve, in milliseconds.
        `tau_minus`
            time constant of the negative part of the STDP curve, in milliseconds.
        `A_plus`:
            amplitude of the positive part of the STDP curve.
        `A_minus`:
            amplitude of the negative part of the STDP curve.
    """

    default_parameters = {
        'tau_plus':  20.0,
        'tau_minus': 20.0,
        'A_plus' : 0.01,
        'A_minus': 0.01,
    }

    def __init__(self, tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.01):
        """
        Create a new specification for the timing-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        STDPTimingDependence.__init__(self, **parameters)


class Vogels2011Rule(STDPTimingDependence):
    """
    Timing-dependence rule from 

      Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011)
      Inhibitory plasticity balances excitation and inhibition in sensory
      pathways and memory networks. Science 334:1569-73
      http://dx.doi.org/10.1126/science.1211095
  
    Potentiation depends on the coincidence of pre- and post-synaptic spikes
    but not on their order. Pre-synaptic spikes in the absence of post-
    synaptic ones produce depression.
    
    Also see http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=143751
    """

    default_parameters = {
        'tau': 20.0,
        'eta': 1e-10,
        'rho': 3.0
    }
