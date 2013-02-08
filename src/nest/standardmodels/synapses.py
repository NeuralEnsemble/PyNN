"""
Synapse Dynamics classes for nest

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import nest
from pyNN.standardmodels import synapses, build_translations
from pyNN.nest.synapses import get_defaults
from pyNN.nest.simulator import state
import logging

logger = logging.getLogger("PyNN")


class StaticSynapse(synapses.StaticSynapse):

    translations = build_translations(
        ('weight', 'weight', 1000.0),
        ('delay', 'delay')
    )
    nest_name = 'static_synapse'

    def _get_nest_synapse_model(self, suffix):
        synapse_defaults = get_defaults(self.nest_name)
        synapse_parameters = self.native_parameters
        for name, value in synapse_parameters.items():
            if value.is_homogeneous:
                value.shape = (1,)
                synapse_defaults[name] = value.evaluate(simplify=True)
        synapse_defaults.pop("tau_minus")
        label = "%s_%s" % (self.nest_name, suffix)
        nest.CopyModel(self.nest_name, label, synapse_defaults)
        return label

    def _get_minimum_delay(self):
        return state.min_delay


#class ComposedSynapseType(ComposedSynapseType):
#    __doc__ = ComposedSynapseType.__doc__
#
#    def _get_nest_synapse_model(self, suffix):
#        # We create a particular synapse context for each projection, by copying
#        # the one which is desired.
#        if self.fast:
#            if self.slow:
#                raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
#            else:
#                base_model = self.fast.native_name
#        elif self.slow:
#            base_model = self.slow.possible_models
#            if isinstance(base_model, set):
#                logger.warning("Several STDP models are available for these connections:")
#                logger.warning(", ".join(model for model in base_model))
#                base_model = list(base_model)[0]
#                logger.warning("By default, %s is used" % base_model)
#        else:
#            base_model = "static_synapse"
#        available_models = nest.Models(mtype='synapses')
#        if base_model not in available_models:
#            raise ValueError("Synapse dynamics model '%s' not a valid NEST synapse model. "
#                             "Possible models in your NEST build are: %s" % (base_model, available_models))
#
#        # CopyModel defaults must be simple floats, so we use the NEST defaults
#        # for any inhomogeneous parameters, and set the inhomogeneous values
#        # later
#        synapse_defaults = get_defaults(base_model)
#        synapse_parameters = self.native_parameters
#        for name, value in synapse_parameters.items():
#            if value.is_homogeneous:
#                value.shape = (1,)
#                synapse_defaults[name] = value.evaluate(simplify=True)
#        # Tau_minus is a parameter of the post-synaptic cell, not of the connection
#        synapse_defaults.pop("tau_minus")
#        label = "%s_%s" % (base_model, suffix)
#        nest.CopyModel(base_model, label, synapse_defaults)
#        return label
#
#    def _set_tau_minus(self, cells):
#        if len(cells) > 0 and self.slow:
#            if 'tau_minus' in nest.GetStatus([cells[0]])[0]:
#                native_parameters = self.slow.timing_dependence.native_parameters
#                if not native_parameters["tau_minus"].is_homogeneous: # could allow inhomogeneous values as long as each column is internally homogeneous
#                    raise ValueError("pyNN.NEST does not support tau_minus being different for different synapses")
#                native_parameters.size = 1 # hack
#                tau_minus = native_parameters["tau_minus"].evaluate(simplify=True)
#                nest.SetStatus(cells.tolist(), [{'tau_minus': tau_minus}])
#            else:
#                raise Exception("Postsynaptic cell model does not support STDP.")
#
#    def _get_minimum_delay(self):
#        return state.min_delay


class STDPMechanism(synapses.STDPMechanism):
    """Specification of STDP models."""

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        assert dendritic_delay_fraction == 1, """NEST does not currently support axonal delays:
                                                 for the purpose of STDP calculations all delays
                                                 are assumed to be dendritic."""
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction)


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight', 1000.0),
        ('delay', 'delay'),
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_fac'),
        ('u0', 'u'),  # this could cause problems for reverse translation
        ('x0', 'x' ), # (as for V_m) in cell models, since the initial value
        ('y0', 'y')   # is not stored, only set.
    )
    native_name = 'tsodyks_synapse'


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 0.0
    }

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 1.0,
        'mu_minus': 1.0
    }
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 1.0
    }

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
        ('mu_plus',   'mu_plus'),
        ('mu_minus',  'mu_minus'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])

    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01, mu_plus=0.5, mu_minus=0.5):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        synapses.GutigWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tau_plus'),
        ('tau_minus', 'tau_minus'), # defined in post-synaptic neuron
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
