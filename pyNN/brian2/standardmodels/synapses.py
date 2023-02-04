# encoding: utf-8
"""
Standard cells for the Brian2 module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from brian2 import ms
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
            ('weight', 'weight'),
            ('delay', 'delay', ms))

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    def _set_target_type(self, weight_units):
        self.translations["weight"]["forward_transform"] = lambda **P: P["weight"] * weight_units
        self.translations["weight"]["reverse_transform"] = lambda **P: P["weight"] / weight_units


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay', ms),
        ('U', 'U'),
        ('tau_rec', 'tau_rec', ms),
        ('tau_facil', 'tau_facil', ms),
        ('tau_syn', 'tau_syn', ms)
    )
    eqs = '''weight : %(weight_units)s
            U : 1
            tau_syn : second
            tau_rec : second
            tau_facil : second
            dz/dt = z/tau_rec : 1 (event-driven)
            dy/dt = -y/tau_syn  : 1 (event-driven)
            du/dt = -u/tau_facil : 1 (event-driven)
            x=1-y-z : 1

            '''
    pre = '''

            u +=  U*(1-u)
            u = int(u > U)*U + int(u <= U)*u
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
        self.translations["weight"]["forward_transform"] = lambda **P: P["weight"] * weight_units
        self.translations["weight"]["reverse_transform"] = lambda **P: P["weight"] / weight_units


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay', ms),
        ('dendritic_delay_fraction', 'dendritic_delay_fraction')
    )
    eqs = """weight : %(weight_units)s
             tau_plus : second
             tau_minus : second
             w_max : %(weight_units)s
             w_min : %(weight_units)s
             A_plus : 1
             A_minus : 1
             dP/dt = -P/tau_plus : 1 (event-driven)
             dM/dt = -M/tau_minus : 1 (event-driven)"""  # to be split among component parts
    pre = """
          P += A_plus
          weight = weight + w_max * M
          weight = int(weight >= w_min)*weight + int(weight < w_min)*w_min
          %(syn_var)s += weight
          """
    post = """
           M -= A_minus
           weight = weight + w_max * P
           weight = int(weight > w_max)*w_max + int(weight <= w_max)*weight
           """
    # for consistency with NEST, the synaptic variable is only updated on a pre-synaptic spike
    initial_conditions = {"M": 0.0, "P": 0.0}

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        if dendritic_delay_fraction != 0:
            raise ValueError("The pyNN.brian2 backend does not currently support "
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
        self.translations["weight"]["forward_transform"] = lambda **P: P["weight"] * weight_units
        self.translations["weight"]["reverse_transform"] = lambda **P: P["weight"] / weight_units
        self.weight_dependence._set_target_type(weight_units)


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'w_max'),
        ('w_min', 'w_min'),
    )

    def _set_target_type(self, weight_units):
        self.translations["w_max"]["forward_transform"] = lambda **P: P["w_max"] * weight_units
        self.translations["w_max"]["reverse_transform"] = lambda **P: P["w_max"] / weight_units
        self.translations["w_min"]["forward_transform"] = lambda **P: P["w_min"] * weight_units
        self.translations["w_min"]["reverse_transform"] = lambda **P: P["w_min"] / weight_units


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'w_max'),
        ('w_min', 'w_min'),
    )

    def _set_target_type(self, weight_units):
        self.translations["w_max"]["forward_transform"] = lambda **P: P["w_max"] * weight_units
        self.translations["w_max"]["reverse_transform"] = lambda **P: P["w_max"] / weight_units
        self.translations["w_min"]["forward_transform"] = lambda **P: P["w_min"] * weight_units
        self.translations["w_min"]["reverse_transform"] = lambda **P: P["w_min"] / weight_units


class AdditivePotentiationMultiplicativeDepression(
    synapses.AdditivePotentiationMultiplicativeDepression
):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max', 'w_max'),
        ('w_min', 'w_min'),
    )

    def _set_target_type(self, weight_units):
        self.translations["w_max"]["forward_transform"] = lambda **P: P["w_max"] * weight_units
        self.translations["w_max"]["reverse_transform"] = lambda **P: P["w_max"] / weight_units
        self.translations["w_min"]["forward_transform"] = lambda **P: P["w_min"] * weight_units
        self.translations["w_min"]["reverse_transform"] = lambda **P: P["w_min"] / weight_units


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max', 'w_max'),
        ('w_min', 'w_min'),
        ('mu_plus', 'mu_plus'),
        ('mu_minus', 'mu_minus'),
    )

    def _set_target_type(self, weight_units):
        self.translations["w_max"]["forward_transform"] = lambda **P: P["w_max"] * weight_units
        self.translations["w_max"]["reverse_transform"] = lambda **P: P["w_max"] / weight_units
        self.translations["w_min"]["forward_transform"] = lambda **P: P["w_min"] * weight_units
        self.translations["w_min"]["reverse_transform"] = lambda **P: P["w_min"] / weight_units


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('A_plus',    'A_plus'),
        ('A_minus',   'A_minus'),
        ('tau_plus', 'tau_plus', ms),
        ('tau_minus', 'tau_minus', ms),
    )
