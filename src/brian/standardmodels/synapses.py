"""
Synapse Dynamics classes for the brian module.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from pyNN.standardmodels import build_translations, synapses, SynapseDynamics, STDPMechanism
from brian import ms

class STDPMechanism(STDPMechanism):
    """Specification of STDP models."""

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        assert dendritic_delay_fraction == 0, """Brian does not currently support dendritic delays:
                                                 for the purpose of STDP calculations all delays
                                                 are assumed to be axonal."""
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction)


class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    __doc__ = synapses.TsodyksMarkramMechanism.__doc__

    translations = build_translations(
        ('U', 'U'),
        ('tau_rec', 'tau_rec', ms),
        ('tau_facil', 'tau_facil', ms),
        ('u0', 'u0'), 
        ('x0', 'x0' ),
        ('y0', 'y0')
    )

    def reset(population, spikes, v_reset):
        population.R_[spikes] -= U_SE*population.R_[spikes]
        population.v_[spikes]= v_reset


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 0.0
    }


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__
    extra_parameters = {
        'mu_plus': 1.0,
        'mu_minus': 1.0
    }


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__
    extra_parameters = {
        'mu_plus': 0.0,
        'mu_minus': 1.0
    }


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__
