"""
Synapse Dynamics classes for the neuron module.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from pyNN.standardmodels import synapses, build_translations, STDPMechanism
from pyNN.neuron.simulator import state


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )
    model = None

    def _get_minimum_delay(self):
        return state.min_delay


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay'),
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_facil'),
        ('u0', 'u0'),
        ('x0', 'x' ), # } note that these two values
        ('y0', 'y')   # } are not used
    )
    model = 'TsodyksMarkramWA'
    postsynaptic_variable = None

    def __init__(self, weight=0.0, delay=None, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        assert (x0 == 1 and y0 == 0), "It is not currently possible to set x0 and y0"
        synapses.TsodyksMarkramSynapse.__init__(self, weight=weight, delay=delay,
                                                U=U, tau_rec=tau_rec,
                                                tau_facil=tau_facil, u0=u0,
                                                x0=x0, y0=y0)


class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    __doc__ = synapses.TsodyksMarkramMechanism.__doc__

    translations = build_translations(
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_facil'),
        ('u0', 'u0'),
        ('x0', 'x' ), # } note that these two values
        ('y0', 'y')   # } are not used
    )
    native_name = 'tsodkys-markram'

    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        assert (x0 == 1 and y0 == 0), "It is not currently possible to set x0 and y0"
        synapses.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSA',])


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSoft',])


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaGuetig'])
    extra_parameters = {
        'muLTP': 0.0,
        'muLTD': 1.0
    }


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
        ('mu_plus',   'muLTP'),
        ('mu_minus',  'muLTD'),
    )
    possible_models = set(['StdwaGuetig'])


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
    )
    possible_models = set(['StdwaSA', 'StdwaSoft', 'StdwaGuetig'])
