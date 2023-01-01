"""
Synapse Dynamics classes for the Arbor module.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import synapses, build_translations
# from pyNN.arbor.simulator import state, Connection, GapJunction, GapJunctionPresynaptic
from pyNN.arbor.simulator import state
from pyNN.arbor.projections import Connection
from pyNN.standardmodels import ion_channels as standard, build_translations


class BaseSynapse(object):
    """
    Base synapse type for all NEURON standard synapses (sets a default 'connection_type')
    """
    connection_type = Connection
    presynaptic_type = None


class StaticSynapse(BaseSynapse, synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )
    model = None

    def _get_minimum_delay(self):
        return state.min_delay


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('U', 'UU'),
        ('tau_rec', 'TAU_REC'),
        ('tau_facil', 'TAU_FACIL'),
        ('u0', 'U0'),
        ('x0', 'X'),
        ('y0', 'Y')
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('dendritic_delay_fraction', 'dendritic_delay_fraction')
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'wmax'),
        ('w_min', 'wmin'),
        ('A_plus', 'aLTP'),
        ('A_minus', 'aLTD'),
    )


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'wmax'),
        ('w_min', 'wmin'),
        ('A_plus', 'aLTP'),
        ('A_minus', 'aLTD'),
    )


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max', 'wmax'),
        ('w_min', 'wmin'),
        ('A_plus', 'aLTP'),
        ('A_minus', 'aLTD'),
    )


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'wmax'),
        ('w_min', 'wmin'),
        ('A_plus', 'aLTP'),
        ('A_minus', 'aLTD'),
        ('mu_plus', 'muLTP'),
        ('mu_minus', 'muLTD'),
    )


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus', 'tauLTP'),
        ('tau_minus', 'tauLTD'),
        ('A_plus', 'aLTP'),
        ('A_minus', 'aLTD'),
    )


# class CondExpPostSynapticResponse(standard.CondExpPostSynapticResponse):
#     """
#     Synapse with discontinuous change in conductance at an event followed by an exponential decay with time constant tau.
#     """
#     translations = build_translations(
#         ('density', 'density'),
#         ('e_rev', 'e'),
#         ('tau_syn', 'tau')
#     )
#     model = "expsyn"
