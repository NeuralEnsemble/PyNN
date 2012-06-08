# encoding: utf-8
"""
Standard cells for the mock module.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import cells, build_translations
import logging

logger = logging.getLogger("PyNN")


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__

    translations = build_translations(  # should add some computed/scaled parameters
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )

class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__

    translations = build_translations(  # should add some computed/scaled parameters
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'T_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )


class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )
    
class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )


class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'GBAR_NA'),
        ('gbar_K',     'GBAR_K'),
        ('g_leak',     'G_LEAK'),
        ('cm',         'CM'),
        ('v_offset',   'V_OFFSET'),
        ('e_rev_Na',   'E_REV_NA'),
        ('e_rev_K',    'E_REV_K'),
        ('e_rev_leak', 'E_REV_LEAK'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('i_offset',   'I_OFFSET'),
    )

class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'SPIKE_TIMES'),
    )

class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'CM'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('v_spike',    'V_SPIKE'),
        ('v_reset',    'V_RESET'),
        ('v_rest',     'V_REST'),
        ('tau_m',      'TAU_M'),
        ('i_offset',   'I_OFFSET'),
        ('a',          'A'),
        ('b',          'B'),
        ('delta_T',    'DELTA_T'),
        ('tau_w',      'TAU_W'),
        ('v_thresh',   'V_THRESH'),
        ('e_rev_E',    'E_REV_E'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )