# encoding: utf-8
"""
Standard cells for the Brian module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from brian import ms, uS, nA
from pyNN.standardmodels import synapses, build_translations
from ..simulator import state


logger = logging.getLogger("PyNN")


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    
    translations = None
    eqs = {"current":     """weight : nA""",
           "conductance": """weight:  uS"""}
    pre = "%s += weight"
    post = None

    def _get_minimum_delay(self):
        return state.min_delay
    
    def _set_target_type(self, target_type):
        if target_type  is None:
            self.translations = None
        elif target_type == "current":
            self.translations = build_translations(
                ('weight', 'weight', nA),
                ('delay', 'delay', ms),
            )
        elif target_type == "conductance":
            self.translations = build_translations(
                ('weight', 'weight', uS),
                ('delay', 'delay', ms),
            )
        else:
            raise ValueError("Only current-based and conductance-based synapses currently supported. You asked for %s" % target_type)

from numpy import exp
class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = None
    eqs = {"current": '''weight : nA
                         dx/dt = (1-x)/tau_rec : 1 (event-driven)
                         du/dt = (U-u)/tau_facil : 1 (event-driven)
                         U : 1
                         tau_rec : ms
                         tau_facil : ms
                         u0 : 1
                         x0 : 1
                         y0 : 1''',
           "conductance": '''weight : uS
                             dx/dt = (1-x)/tau_rec : 1 (event-driven)
                             du/dt = (U-u)/tau_facil : 1 (event-driven)
                             U : 1
                             tau_rec : ms
                             tau_facil : ms
                             u0 : 1
                             x0 : 1
                             y0 : 1'''}
    pre = '''%s += weight*u*x
             x *= (1-u)
             u += U*(1-u)'''
    post = None
    
    def _get_minimum_delay(self):
        return state.min_delay

    def _set_target_type(self, target_type):
        if target_type  is None:
            self.translations = None
        elif target_type == "current":
            self.translations = build_translations(
                ('weight', 'weight', nA),
                ('delay', 'delay', ms),
                ('U', 'U'),
                ('tau_rec', 'tau_rec', ms),
                ('tau_facil', 'tau_facil', ms),
                ('u0', 'u0'),   # unused, arguably should be moved to initial conditions
                ('x0', 'x0' ),  # unused
                ('y0', 'y0')    # unused
            )
        elif target_type == "conductance":
            self.translations = build_translations(
                ('weight', 'weight', uS),
                ('delay', 'delay', ms),
                ('U', 'U'),
                ('tau_rec', 'tau_rec', ms),
                ('tau_facil', 'tau_facil', ms),
                ('u0', 'u0'),   # unused
                ('x0', 'x0' ),  # unused
                ('y0', 'y0')    # unused
            )
        else:
            raise ValueError("Only current-based and conductance-based synapses currently supported. You asked for %s" % target_type)


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
