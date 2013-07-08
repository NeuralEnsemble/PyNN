"""
Synapse Dynamics classes for the neuron module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import synapses, build_translations
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


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__
    postsynaptic_variable = 'spikes'

    base_translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay')
    )  # will be extended by translations from timing_dependence, etc.

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        super(STDPMechanism, self).__init__(timing_dependence,
                                            weight_dependence,
                                            voltage_dependence,
                                            dendritic_delay_fraction,
                                            weight, delay)
        if dendritic_delay_fraction > 0.5 and state.num_processes > 1:
            # depending on delays, can run into problems with the delay from the
            # pre-synaptic neuron to the weight-adjuster mechanism being zero.
            # The best (only?) solution would be to create connections on the
            # node with the pre-synaptic neurons for ddf>0.5 and on the node
            # with the post-synaptic neuron (as is done now) for ddf<0.5
            raise NotImplementedError("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")

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

    def _get_minimum_delay(self):
        return state.min_delay
    
    
class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
    )
    possible_models = set(['StdwaSA', 'StdwaVogels2011'])


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
    )
    possible_models = set(['StdwaSoft',])


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
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
        ('mu_plus',   'muLTP'),
        ('mu_minus',  'muLTD'),
    )
    possible_models = set(['StdwaGuetig'])


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),

    )
    possible_models = set(['StdwaSA', 'StdwaSoft', 'StdwaGuetig'])


class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__
    
    translations = build_translations(
        ('tau',  'tauLTP'),
        ('eta', 'eta'),
        ('rho', 'rho'),
    )
    possible_models = set(['StdwaVogels2011'])
