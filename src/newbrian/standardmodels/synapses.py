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
    initial_conditions = {}

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


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = None
    eqs = {"current": '''weight : nA
                             u : 1
                             x : 1
                             y : 1
                             z : 1
                             U : 1
                             tau_syn : ms
                             tau_rec : ms
                             tau_facil : ms''',
           "conductance": '''weight : uS
                             u : 1
                             x : 1
                             y : 1
                             z : 1
                             U : 1
                             tau_syn : ms
                             tau_rec : ms
                             tau_facil : ms'''}
    pre = '''z *= exp(-(t - lastupdate)/tau_rec)
             z += y*(exp(-(t - lastupdate)/tau_syn) - exp(-(t - lastupdate)/tau_rec)) / ((tau_syn/tau_rec) - 1)
             y *= exp(-(t - lastupdate)/tau_syn)
             x = 1 - y - z
             u *= exp(-(t - lastupdate)/tau_facil)
             u = max(u + U*(1-u), U)
             %s += weight*x*u
             y += x*u
             '''
    post = None
    initial_conditions = {"u": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    tau_syn_var = {"excitatory": "tau_syn_E",
                   "inhibitory": "tau_syn_I"}
    
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
            )
        elif target_type == "conductance":
            self.translations = build_translations(
                ('weight', 'weight', uS),
                ('delay', 'delay', ms),
                ('U', 'U'),
                ('tau_rec', 'tau_rec', ms),
                ('tau_facil', 'tau_facil', ms),
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
    initial_conditions = {"M": 0.0, "P": 0.0}

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
