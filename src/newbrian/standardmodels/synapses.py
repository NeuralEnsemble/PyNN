# encoding: utf-8
"""
Standard cells for the Brian module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from brian import ms, uS
from pyNN.standardmodels import synapses, build_translations
from ..simulator import state


logger = logging.getLogger("PyNN")


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    
    translations = build_translations(
        ('weight', 'weight', uS),  # need nA for current-base synapses. How to handle?
        ('delay', 'delay', ms),
    )
    eqs = """weight : uS"""   # units should depend on whether current or conductance based
    pre = "%s += weight"
    post = None

    def _get_minimum_delay(self):
        return state.min_delay


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('U', 'UU'),
        ('tau_rec', 'TAU_REC'),
        ('tau_facil', 'TAU_FACIL'),
        ('u0', 'U0'),
        ('x0', 'X' ),
        ('y0', 'Y')
    )
    
    def _get_minimum_delay(self):
        return state.min_delay


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('weight', 'weight', uS),
        ('delay', 'delay', ms)
    )
    eqs = """
          weight : uS
          tau_plus : ms
          tau_minus : ms
          w_max : uS
          w_min : uS
          A_plus : 1
          A_minus : 1
          dP/dt = -P/tau_plus : 1 (event-driven)
          dM/dt = -M/tau_minus : 1 (event-driven)
          """
    pre = """
          P += A_plus
          weight = max(weight + w_max * M, w_min)
          %s += weight
          """    
    post = """
           M -= A_minus
           weight = min(weight + w_max * P, w_max)
           %s += weight
           """

    def _get_minimum_delay(self):
        return state.min_delay
    

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max'),
        ('w_min',     'w_min'),
    )


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max'),
        ('w_min',     'w_min'),
    )


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'w_max'),
        ('w_min',     'w_min'),
    )


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max'),
        ('w_min',     'w_min'),
        ('mu_plus',   'mu_plus'),
        ('mu_minus',  'mu_minus'),
    )


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('A_plus',    'A_plus'),
        ('A_minus',   'A_minus'),
        ('tau_plus',  'tau_plus', ms),
        ('tau_minus', 'tau_plus', ms),
    )
