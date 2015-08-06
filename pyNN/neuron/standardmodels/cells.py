# encoding: utf-8
"""
Standard base_cells for the neuron module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import cells as base_cells, build_translations
from pyNN.neuron.cells import (StandardIF, SingleCompartmentTraub,
                               RandomSpikeSource, VectorSpikeSource,
                               BretteGerstnerIF, GsfaGrrIF, Izhikevich_)
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
    model = StandardIF
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
    model = StandardIF
    extra_parameters = {'syn_type': 'current',
                        'syn_shape': 'exp'}


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
    model = StandardIF
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
    model = StandardIF
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
    model = StandardIF
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
        ('a',          'A',        0.001), # nS --> uS
        ('b',          'B'),
        ('delta_T',    'delta'),
        ('tau_w',      'tau_w'),
        ('v_thresh',   'v_thresh'),
        ('e_rev_E',    'e_e'),
        ('tau_syn_E',  'tau_e'),
        ('e_rev_I',    'e_i'),
        ('tau_syn_I',  'tau_i'),
    )
    model = BretteGerstnerIF
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'alpha'}


class EIF_cond_exp_isfa_ista(base_cells.EIF_cond_exp_isfa_ista):

    __doc__ = base_cells.EIF_cond_exp_isfa_ista.__doc__

    translations = EIF_cond_alpha_isfa_ista.translations
    model = BretteGerstnerIF
    extra_parameters = {'syn_type': 'conductance',
                        'syn_shape': 'exp'}


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
