"""
Synapse Dynamics classes for nest

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import nest
from pyNN.standardmodels import synapses, build_translations
from pyNN.nest.synapses import get_synapse_defaults, NESTSynapseMixin
import logging

from ..conversion import make_sli_compatible

logger = logging.getLogger("PyNN")


class StaticSynapse(synapses.StaticSynapse, NESTSynapseMixin):

    translations = build_translations(
        ('weight', 'weight', 1000.0),
        ('delay', 'delay')
    )
    nest_name = 'static_synapse'


class STDPMechanism(synapses.STDPMechanism, NESTSynapseMixin):
    """Specification of STDP models."""
    
    base_translations = build_translations(
        ('weight', 'weight', 1000.0),  # nA->pA, uS->nS
        ('delay', 'delay')
    )  # will be extended by translations from timing_dependence, etc.
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        if dendritic_delay_fraction != 1:
            raise ValueError("NEST does not currently support axonal delays: "
                             "for the purpose of STDP calculations all delays "
                             "are assumed to be dendritic.")
        # could perhaps support axonal delays using parrot neurons?
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction,
                                            weight, delay)

    def _get_nest_synapse_model(self, suffix):
        base_model = self.possible_models
        if isinstance(base_model, set):
            logger.warning("Several STDP models are available for these connections:")
            logger.warning(", ".join(model for model in base_model))
            base_model = list(base_model)[0]
            logger.warning("By default, %s is used" % base_model)
        available_models = nest.Models(mtype='synapses')
        if base_model not in available_models:
            raise ValueError("Synapse dynamics model '%s' not a valid NEST synapse model. "
                             "Possible models in your NEST build are: %s" % (base_model, available_models))

        # CopyModel defaults must be simple floats, so we use the NEST defaults
        # for any inhomogeneous parameters, and set the inhomogeneous values
        # later
        synapse_defaults = {}
        for name, value in self.native_parameters.items():
            if value.is_homogeneous:
                value.shape = (1,)
                synapse_defaults[name] = value.evaluate(simplify=True)
        synapse_defaults.pop("dendritic_delay_fraction")
        synapse_defaults.pop("w_min_always_zero_in_NEST")
        # Tau_minus is a parameter of the post-synaptic cell, not of the connection
        synapse_defaults.pop("tau_minus")

        synapse_defaults = make_sli_compatible(synapse_defaults)

        label = "%s_%s" % (base_model, suffix)
        nest.CopyModel(base_model, label, synapse_defaults)
        return label


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse, NESTSynapseMixin):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight', 1000.0),
        ('delay', 'delay'),
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_fac')
    )
    nest_name = 'tsodyks_synapse'


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 0.0
    }

    def __init__(self, w_min=0.0, w_max=1.0):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.AdditiveWeightDependence.__init__(self, w_min, w_max)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 1.0,
        'mu_minus': 1.0
    }
    def __init__(self, w_min=0.0, w_max=1.0):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.MultiplicativeWeightDependence.__init__(self, w_min, w_max)


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 1.0
    }

    def __init__(self, w_min=0.0, w_max=1.0):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max)


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('mu_plus',   'mu_plus'),
        ('mu_minus',  'mu_minus'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])

    def __init__(self, w_min=0.0, w_max=1.0, mu_plus=0.5, mu_minus=0.5):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.GutigWeightDependence.__init__(self, w_min, w_max)


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tau_plus'),
        ('tau_minus', 'tau_minus'), # defined in post-synaptic neuron
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),

    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
