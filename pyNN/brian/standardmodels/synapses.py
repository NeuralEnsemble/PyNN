# encoding: utf-8
"""
Standard cells for the Brian module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from brian import ms, uS, nA
from pyNN.standardmodels import synapses, build_translations
from ..simulator import state


logger = logging.getLogger("PyNN")


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__

    eqs = """weight : %(weight_units)s"""
    pre = "%(syn_var)s += weight"
    post = None
    initial_conditions = {}

    def __init__(self, **parameters):
        super(StaticSynapse, self).__init__(**parameters)
        # we have to define the translations on a per-instance basis because
        # they depend on whether the synapses are current-, conductance- or voltage-based.
        self.translations = build_translations(
                                ('weight', 'weight', "weight*weight_units", "weight/weight_units"),
                                ('delay', 'delay', ms))

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d
    
    def _set_target_type(self, weight_units):
        for key, value in self.translations.items():
            for direction in ("forward_transform", "reverse_transform"):
                self.translations[key][direction] = value[direction].replace("weight_units", str(float(weight_units)))



class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
                        ('weight', 'weight', "weight*weight_units", "weight/weight_units"),
                        ('delay', 'delay', ms),
                        ('U', 'U'),
                        ('tau_rec', 'tau_rec', ms),
                        ('tau_facil', 'tau_facil', ms),
                   )
    eqs = '''weight : %(weight_units)s
             u : 1
             x : 1
             y : 1
             z : 1
             U : 1
             tau_syn : ms
             tau_rec : ms
             tau_facil : ms'''
    pre = '''z *= exp(-(t - lastupdate)/tau_rec)
             z += y*(exp(-(t - lastupdate)/tau_syn) - exp(-(t - lastupdate)/tau_rec)) / ((tau_syn/tau_rec) - 1)
             y *= exp(-(t - lastupdate)/tau_syn)
             x = 1 - y - z
             u *= exp(-(t - lastupdate)/tau_facil)
             u += U*(1-u)
             mask = u > U
             u[mask] = U[mask]
             %(syn_var)s += weight*x*u
             y += x*u
             '''
    post = None
    initial_conditions = {"u": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    tau_syn_var = {"excitatory": "tau_syn_E",
                   "inhibitory": "tau_syn_I"}
    
    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    def _set_target_type(self, weight_units):
        for key, value in self.translations.items():
            for direction in ("forward_transform", "reverse_transform"):
                self.translations[key][direction] = value[direction].replace("weight_units", str(float(weight_units)))


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
                            ('weight', 'weight', "weight*weight_units", "weight/weight_units"),
                            ('delay', 'delay', ms),
                        )
    eqs = """weight : %(weight_units)s
             tau_plus : ms
             tau_minus : ms
             w_max : %(weight_units)s
             w_min : %(weight_units)s
             A_plus : 1
             A_minus : 1
             dP/dt = -P/tau_plus : 1 (event-driven)
             dM/dt = -M/tau_minus : 1 (event-driven)"""  # to be split among component parts
    pre = """
          P += A_plus
          weight = weight + w_max * M
          mask = weight < w_min
          weight[mask] = w_min[mask]
          %(syn_var)s += weight
          """
    post = """
           M -= A_minus
           weight = weight + w_max * P
           mask = weight > w_max
           weight[mask] = w_max[mask]
           """  # for consistency with NEST, the synaptic variable is only updated on a pre-synaptic spike
    initial_conditions = {"M": 0.0, "P": 0.0}

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        if (dendritic_delay_fraction != 0):
            raise ValueError("The pyNN.brian backend does not currently support "
                             "dendritic delays: for the purpose of STDP calculations "
                             "all delays are assumed to be axonal.")
        # could perhaps support axonal delays using parrot neurons?
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction,
                                            weight, delay)

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    def _set_target_type(self, weight_units):
        for key, value in self.translations.items():
            for direction in ("forward_transform", "reverse_transform"):
                self.translations[key][direction] = value[direction].replace("weight_units", str(float(weight_units)))
    

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max', "w_max*weight_units", "w_max/weight_units"),
        ('w_min',     'w_min', "w_min*weight_units", "w_min/weight_units"),
    )


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max', "w_max*weight_units", "w_max/weight_units"),
        ('w_min',     'w_min', "w_min*weight_units", "w_min/weight_units"),
    )


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'w_max', "w_max*weight_units", "w_max/weight_units"),
        ('w_min',     'w_min', "w_min*weight_units", "w_min/weight_units"),
    )


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'w_max', "w_max*weight_units", "w_max/weight_units"),
        ('w_min',     'w_min', "w_min*weight_units", "w_min/weight_units"),
        ('mu_plus',   'mu_plus'),
        ('mu_minus',  'mu_minus'),
    )


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('A_plus',    'A_plus'),
        ('A_minus',   'A_minus'),
        ('tau_plus',  'tau_plus', ms),
        ('tau_minus', 'tau_minus', ms),
    )
