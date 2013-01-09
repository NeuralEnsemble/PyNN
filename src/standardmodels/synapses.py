# encoding: utf-8
"""
Definition of default parameters (and hence, standard parameter names) for
standard dynamic synapse models.

Classes for specifying short-term plasticity (facilitation/depression):
    TsodyksMarkramMechanism

Classes for defining STDP rules:
    AdditiveWeightDependence
    MultiplicativeWeightDependence
    AdditivePotentiationMultiplicativeDepression
    GutigWeightDependence
    SpikePairRule

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import StandardSynapseType, ShortTermPlasticityMechanism, STDPWeightDependence, STDPTimingDependence


class StaticSynapse(StandardSynapseType):
    """
    Synaptic connection with fixed weight and delay.
    """
    default_parameters = {
        'weight': 0.0,
        'delay': None
    }


class TsodyksMarkramSynapse(StandardSynapseType):
    """
    Synapse exhibiting facilitation and depression, implemented using the model
    of Tsodyks, Markram et al.:

    Tsodyks, Uziel, Markram (2000) Synchrony Generation in Recurrent Networks
       with Frequency-Dependent Synapses. Journal of Neuroscience, vol 20 RC50

    Note that the time constant of the post-synaptic current is set in the
    neuron model, not here.

    Arguments:
        `U`:
            use parameter.
        `tau_rec`:
            depression time constant (ms).
        `tau_facil`:
            facilitation time constant (ms).
        `u0`, `x0`, `y0`:
            initial conditions.
    """
    default_parameters = {
        'weight': 0.0,
        'delay': None,
        'U': 0.5,   # use parameter
        'tau_rec': 100.0, # depression time constant (ms)
        'tau_facil': 0.0,   # facilitation time constant (ms)
        'u0': 0.0,  # }
        'x0': 1.0,  # } initial values
        'y0': 0.0   # }
    }


class TsodyksMarkramMechanism(ShortTermPlasticityMechanism):
    """
    Synapse exhibiting facilitation and depression, implemented using the model
    of Tsodyks, Markram et al.:

    Tsodyks, Uziel, Markram (2000) Synchrony Generation in Recurrent Networks
       with Frequency-Dependent Synapses. Journal of Neuroscience, vol 20 RC50

    Note that the time constant of the post-synaptic current is set in the
    neuron model, not here.

    Arguments:
        `U`:
            use parameter.
        `tau_rec`:
            depression time constant (ms).
        `tau_facil`:
            facilitation time constant (ms).
        `u0`, `x0`, `y0`:
            initial conditions.
    """
    default_parameters = {
        'U': 0.5,   # use parameter
        'tau_rec': 100.0, # depression time constant (ms)
        'tau_facil': 0.0,   # facilitation time constant (ms)
        'u0': 0.0,  # }
        'x0': 1.0,  # } initial values
        'y0': 0.0   # }
    }

    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        """
        Create a new specification for a short-term plasticity mechanism.
        """
        parameters = dict(locals())
        parameters.pop('self')
        ShortTermPlasticityMechanism.__init__(self, **parameters)


class AdditiveWeightDependence(STDPWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.

    Arguments:
        `w_min`:
            minimum synaptic weight, in the same units as the weight, i.e.
            µS or nA.
        `w_max`:
            maximum synaptic weight.
        `A_plus`:
            synaptic weight increase as a fraction of `w_max` when the
            pre-synaptic spike precedes the post-synaptic spike by an
            infinitessimal amount.
        `A_minus`:
            synaptic weight decrease as a fraction of `w_max` when the
            pre-synaptic spike lags the post-synaptic spike by an
            infinitessimal amount.
    """
    default_parameters = {
        'w_min':   0.0,
        'w_max':   1.0,
        'A_plus':  0.01,
        'A_minus': 0.01
    }

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        """
        Create a new specification for the weight-dependence of an STDP rule.
        """
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
        `A_plus`:
            synaptic weight increase as a fraction of `w_max-w` when the
            pre-synaptic spike precedes the post-synaptic spike by an
            infinitessimal amount.
        `A_minus`:
            synaptic weight decrease as a fraction of `w-w_min` when the
            pre-synaptic spike lags the post-synaptic spike by an
            infinitessimal amount.
    """
    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
        'A_plus' : 0.01,
        'A_minus': 0.01,
    }

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        """
        Create a new specification for the weight-dependence of an STDP rule.
        """
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
        `A_plus`:
            synaptic weight increase as a fraction of `w_max` when the
            pre-synaptic spike precedes the post-synaptic spike by an
            infinitessimal amount.
        `A_minus`:
            synaptic weight decrease as a fraction of `w-w_min` when the
            pre-synaptic spike lags the post-synaptic spike by an
            infinitessimal amount.
    """

    default_parameters = {
        'w_min'  : 0.0,
        'w_max'  : 1.0,
        'A_plus' : 0.01,
        'A_minus': 0.01,
    }

    def __init__(self, w_min=0.0,  w_max=1.0, A_plus=0.01, A_minus=0.01):
        """
        Create a new specification for the weight-dependence of an STDP rule.
        """
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
        `A_plus`:
            synaptic weight increase as a fraction of `(w_max-w)^mu_plus`
            when the pre-synaptic spike precedes the post-synaptic spike by an
            infinitessimal amount.
        `A_minus`:
            synaptic weight decrease as a fraction of `(w-w_min)^mu_minus`
            when the pre-synaptic spike lags the post-synaptic spike by an
            infinitessimal amount.
        `mu_plus`:
            see above
        `mu_minus`:
            see above
    """

    default_parameters = {
        'w_min'   : 0.0,
        'w_max'   : 1.0,
        'A_plus'  : 0.01,
        'A_minus' : 0.01,
        'mu_plus' : 0.5,
        'mu_minus': 0.5
    }

    def __init__(self, w_min=0.0,  w_max=1.0, A_plus=0.01, A_minus=0.01, mu_plus=0.5, mu_minus=0.5):
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
    spike pairs, not triplets, etc.

    Arguments:
        `tau_plus`:
            time constant of the positive part of the STDP curve, in milliseconds.
        `tau_minus`
            time constant of the negative part of the STDP curve, in milliseconds.
    """

    default_parameters = {
        'tau_plus':  20.0,
        'tau_minus': 20.0,
    }

    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        """
        Create a new specification for the timing-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        STDPTimingDependence.__init__(self, **parameters)
