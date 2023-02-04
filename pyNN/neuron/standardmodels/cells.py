# encoding: utf-8
"""
Standard base_cells for the neuron module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import ModelNotAvailable, cells as base_cells, build_translations
from pyNN.neuron.cells import (StandardIFStandardReceptors, SingleCompartmentTraub,
                               RandomSpikeSource, VectorSpikeSource,
                               RandomGammaSpikeSource,
                               RandomPoissonRefractorySpikeSource,
                               StandardIF,
                               BretteGerstnerIF,
                               BretteGerstnerIFStandardReceptors, GsfaGrrIF, Izhikevich_,
                               GIFNeuron)
import logging

logger = logging.getLogger("PyNN")


class IF_curr_alpha(base_cells.IF_curr_alpha):

    __doc__ = base_cells.IF_curr_alpha.__doc__

    translations = build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'c_m'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
    )
    model = StandardIFStandardReceptors
    extra_parameters = {'syn_type': 'current',
                        'syn_shape': 'alpha'}


class IF_curr_exp(base_cells.IF_curr_exp):

    __doc__ = base_cells.IF_curr_exp.__doc__

    translations = build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'c_m'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
    )
    model = StandardIFStandardReceptors
    extra_parameters = {'syn_type': 'current',
                        'syn_shape': 'exp'}


class IF_curr_delta(ModelNotAvailable):
    pass


class IF_cond_alpha(base_cells.IF_cond_alpha):

    __doc__ = base_cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'c_m'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i')
    )
    model = StandardIFStandardReceptors
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'alpha'}


class IF_cond_exp(base_cells.IF_cond_exp):

    __doc__ = base_cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'c_m'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 't_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i')
    )
    model = StandardIFStandardReceptors
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


class IF_facets_hardware1(base_cells.IF_facets_hardware1):

    __doc__ = base_cells.IF_facets_hardware1.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('g_leak',     'tau_m',    "0.2*1000.0/g_leak", "0.2*1000.0/tau_m"),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('e_rev_I',    'e_i')
    )
    model = StandardIFStandardReceptors
    extra_parameters = {'syn_type':  'conductance',
                        'syn_shape': 'exp',
                        'i_offset':  0.0,
                        'c_m':       0.2,
                        't_refrac':  1.0,
                        'e_e':       0.0}


class HH_cond_exp(base_cells.HH_cond_exp):

    __doc__ = base_cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'gbar_Na',   1e-3),   # uS -> mS
        ('gbar_K',     'gbar_K',    1e-3),
        ('g_leak',     'g_leak',    1e-3),
        ('cm',         'c_m'),
        ('v_offset',   'v_offset'),
        ('e_rev_Na',   'ena'),
        ('e_rev_K',    'ek'),
        ('e_rev_leak', 'e_leak'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('i_offset',   'i_offset'),
    )
    model = SingleCompartmentTraub
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


class IF_cond_exp_gsfa_grr(base_cells.IF_cond_exp_gsfa_grr):

    __doc__ = base_cells.IF_cond_exp_gsfa_grr.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_reset',    'v_reset'),
        ('cm',         'c_m'),
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E',  'tau_e'),
        ('tau_syn_I',  'tau_i'),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset'),
        ('e_rev_E',    'e_e'),
        ('e_rev_I',    'e_i'),
        ('tau_sfa',    'tau_sfa'),
        ('e_rev_sfa',  'e_sfa'),
        ('q_sfa',      'q_sfa'),
        ('tau_rr',     'tau_rr'),
        ('e_rev_rr',   'e_rr'),
        ('q_rr',       'q_rr')
    )
    model = GsfaGrrIF
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


class SpikeSourcePoisson(base_cells.SpikeSourcePoisson):

    __doc__ = base_cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'start'),
        ('rate',     '_interval',  "1000.0/rate",  "1000.0/_interval"),
        ('duration', 'duration'),
    )
    model = RandomSpikeSource


class SpikeSourcePoissonRefractory(base_cells.SpikeSourcePoissonRefractory):

    __doc__ = base_cells.SpikeSourcePoissonRefractory.__doc__

    translations = build_translations(
        ('start',      'start'),
        ('rate',       'rate'),
        ('tau_refrac', 'tau_refrac'),
        ('duration',   'duration'),
    )
    model = RandomPoissonRefractorySpikeSource


class SpikeSourceGamma(base_cells.SpikeSourceGamma):
    __doc__ = base_cells.SpikeSourceGamma.__doc__

    translations = build_translations(
        ('alpha',    'alpha'),
        ('beta',     'beta',    0.001),
        ('start',    'start'),
        ('duration', 'duration'),
    )
    model = RandomGammaSpikeSource


class SpikeSourceArray(base_cells.SpikeSourceArray):

    __doc__ = base_cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )
    model = VectorSpikeSource


class EIF_cond_alpha_isfa_ista(base_cells.EIF_cond_alpha_isfa_ista):

    __doc__ = base_cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'c_m'),
        ('tau_refrac', 't_refrac'),
        ('v_spike',    'v_spike'),
        ('v_reset',    'v_reset'),
        ('v_rest',     'v_rest'),
        ('tau_m',      'tau_m'),
        ('i_offset',   'i_offset'),
        ('a',          'A',        0.001),  # nS --> uS
        ('b',          'B'),
        ('delta_T',    'delta'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'v_thresh'),
        ('e_rev_E',    'e_e'),
        ('tau_syn_E',  'tau_e'),
        ('e_rev_I',    'e_i'),
        ('tau_syn_I',  'tau_i'),
    )
    model = BretteGerstnerIFStandardReceptors
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'alpha'}


class EIF_cond_exp_isfa_ista(base_cells.EIF_cond_exp_isfa_ista):
    __doc__ = base_cells.EIF_cond_exp_isfa_ista.__doc__

    translations = EIF_cond_alpha_isfa_ista.translations
    model = BretteGerstnerIFStandardReceptors
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


class LIF(base_cells.LIF):

    translations = build_translations(
        ('cm',         'c_m'),
        ('tau_refrac', 't_refrac'),
        ('v_reset',    'v_reset'),
        ('v_rest',     'v_rest'),
        ('tau_m',      'tau_m'),
        ('i_offset',   'i_offset'),
        ('v_thresh',   'v_thresh'),
    )
    model = StandardIF
    variable_map = {"v": "v"}


class AdExp(base_cells.AdExp):

    translations = build_translations(
        ('cm',         'c_m'),
        ('tau_refrac', 't_refrac'),
        ('v_spike',    'v_spike'),
        ('v_reset',    'v_reset'),
        ('v_rest',     'v_rest'),
        ('tau_m',      'tau_m'),
        ('i_offset',   'i_offset'),
        ('a',          'A',        0.001),  # nS --> uS
        ('b',          'B'),
        ('delta_T',    'delta'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'v_thresh'),
    )
    model = BretteGerstnerIF
    variable_map = {"v": "v", "w": "w"}


class Izhikevich(base_cells.Izhikevich):
    __doc__ = base_cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',        'a_'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'i_offset')
    )
    model = Izhikevich_


class GIF_cond_exp(base_cells.GIF_cond_exp):
    translations = build_translations(
        ('v_rest', 'v_rest'),
        ('cm', 'c_m'),
        ('tau_m', 'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E', 'tau_e'),
        ('tau_syn_I', 'tau_i'),
        ('e_rev_E', 'e_e'),
        ('e_rev_I', 'e_i'),
        ('v_reset', 'v_reset'),
        ('i_offset', 'i_offset'),
        ('delta_v', 'dV'),
        ('v_t_star', 'vt_star'),
        ('lambda0', 'lambda0'),
        ('tau_eta', 'tau_eta'),
        ('tau_gamma', 'tau_gamma'),
        ('a_eta', 'a_eta'),
        ('a_gamma', 'a_gamma'),
    )
    model = GIFNeuron
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


class PointNeuron(base_cells.PointNeuron):

    def __init__(self, neuron, **post_synaptic_receptors):
        super(PointNeuron, self).__init__(neuron, **post_synaptic_receptors)
        self.model = neuron.model

    @property
    def native_parameters(self):
        translated_parameters = self.neuron.native_parameters
        for name, psr in self.post_synaptic_receptors.items():
            translated_parameters.add_child(name, psr.native_parameters)
        return translated_parameters

    def get_native_names(self, *names):
        neuron_names = self.neuron.get_native_names(*[name for name in names if "." not in name])
        for name in names:
            if "." in name:
                psr_name, param_name = name.split(".")
                index = self.receptor_types.index(psr_name)
                tr_name = self.post_synaptic_receptors[psr_name].get_native_names(param_name)
                neuron_names.append("{}[{}]".format(tr_name, index))
        return neuron_names

    def reverse_translate(self, native_parameters):
        standard_parameters = self.neuron.reverse_translate(native_parameters)
        return standard_parameters

    @property
    def variable_map(self):
        var_map = self.neuron.variable_map.copy()
        for name, psr in self.post_synaptic_receptors.items():
            for variable, translated_variable in psr.variable_map.items():
                var_map[f"{name}.{variable}"] = translated_variable
        return var_map
