# encoding: utf-8
"""
Standard cells for the Brian module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from copy import deepcopy
import brian
from brian import mV, ms, nF, nA, uS, Hz, nS
from pyNN.standardmodels import cells, build_translations
from ..simulator import state
from ..cells import (ThresholdNeuronGroup, SpikeGeneratorGroup, PoissonGroup,
                     BiophysicalNeuronGroup, AdaptiveNeuronGroup, AdaptiveNeuronGroup2,
                     IzhikevichNeuronGroup)
import logging

logger = logging.getLogger("PyNN")


leaky_iaf = brian.Equations('''
                dv/dt = (v_rest-v)/tau_m + (i_syn + i_offset + i_inj)/c_m  : mV
                tau_m                   : ms
                c_m                     : nF
                v_rest                  : mV
                i_offset                : nA
                i_inj                   : nA
            ''')

adexp_iaf = brian.Equations('''
                dv/dt = ((v_rest-v) + delta_T*exp((v - v_thresh)/delta_T))/tau_m + (i_syn + i_offset + i_inj - w)/c_m : mV
                dw/dt = (a*(v-v_rest) - w)/tau_w  : nA
                a                       : uS
                tau_m                   : ms
                tau_w                   : ms
                c_m                     : nF
                v_rest                  : mV
                v_thresh                : mV
                delta_T                 : mV
                i_offset                : nA
                i_inj                   : nA
            ''')

# g_r, g_s should be in uS for PyNN unit system consistency
adapt_iaf = brian.Equations('''
                dv/dt = (v_rest-v)/tau_m + (-g_r*(v-E_r) - g_s*(v-E_s) + i_syn + i_offset + i_inj)/c_m  : mV
                dg_s/dt = -g_s/tau_s    : nS
                dg_r/dt = -g_r/tau_r    : nS
                tau_m                   : ms
                tau_s                   : ms
                tau_r                   : ms
                c_m                     : nF
                v_rest                  : mV
                i_offset                : nA
                i_inj                   : nA
                E_r                     : mV
                E_s                     : mV
            ''')\


conductance_based_exponential_synapses = brian.Equations('''
                dge/dt = -ge/tau_syn_e  : uS
                dgi/dt = -gi/tau_syn_i  : uS
                i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)  : nA
                tau_syn_e               : ms
                tau_syn_i               : ms
                e_rev_e                 : mV
                e_rev_i                 : mV
            ''')

conductance_based_alpha_synapses = brian.Equations('''
                dge/dt = (2.7182818284590451*ye-ge)/tau_syn_e  : uS
                dye/dt = -ye/tau_syn_e                         : uS
                dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_i  : uS
                dyi/dt = -yi/tau_syn_i                         : uS
                i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)    : nA
                tau_syn_e               : ms
                tau_syn_i               : ms
                e_rev_e                 : mV
                e_rev_i                 : mV
        ''')

current_based_exponential_synapses = brian.Equations('''
                die/dt = -ie/tau_syn_e  : nA
                dii/dt = -ii/tau_syn_i  : nA
                i_syn = ie + ii         : nA
                tau_syn_e               : ms
                tau_syn_i               : ms
            ''')

current_based_alpha_synapses = brian.Equations('''
                die/dt = (2.7182818284590451*ye-ie)/tau_syn_e : nA
                dye/dt = -ye/tau_syn_e                        : nA
                dii/dt = (2.7182818284590451*yi-ii)/tau_syn_e : nA
                dyi/dt = -yi/tau_syn_e                        : nA
                i_syn = ie + ii                               : nA
                tau_syn_e                                     : ms
                tau_syn_i                                     : ms
            ''')

leaky_iaf_translations = build_translations(
                ('v_rest',     'v_rest',     mV),
                ('v_reset',    'v_reset',    mV),
                ('cm',         'c_m',        nF),
                ('tau_m',      'tau_m',      ms),
                ('tau_refrac', 'tau_refrac', ms),
                ('v_thresh',   'v_thresh',   mV),
                ('i_offset',   'i_offset',   nA))

adexp_iaf_translations = build_translations(
                ('v_rest',     'v_rest',     mV),
                ('v_reset',    'v_reset',    mV),
                ('cm',         'c_m',        nF),
                ('tau_m',      'tau_m',      ms),
                ('tau_refrac', 'tau_refrac', ms),
                ('v_thresh',   'v_thresh',   mV),
                ('i_offset',   'i_offset',   nA),
                ('a',          'a',          nA),
                ('b',          'b',          nA),
                ('delta_T',    'delta_T',    mV),
                ('tau_w',      'tau_w',      ms),
                ('v_spike',    'v_spike',    mV))

adapt_iaf_translations = build_translations(
                ('v_rest',     'v_rest',     mV),
                ('v_reset',    'v_reset',    mV),
                ('cm',         'c_m',        nF),
                ('tau_m',      'tau_m',      ms),
                ('tau_refrac', 'tau_refrac', ms),
                ('v_thresh',   'v_thresh',   mV),
                ('i_offset',   'i_offset',   nA),
                ('tau_sfa',    'tau_s',      ms),
                ('e_rev_sfa',  'E_s',        mV),
                ('q_sfa',      'q_s',        nS),   # should we uS for consistency of PyNN unit system?
                ('tau_rr',     'tau_r',      mV),
                ('e_rev_rr',   'E_r',        mV),
                ('q_rr',       'q_r',        nS))

conductance_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  ms),
                ('tau_syn_I',  'tau_syn_i',  ms),
                ('e_rev_E',    'e_rev_e',    mV),
                ('e_rev_I',    'e_rev_i',    mV))

current_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  ms),
                ('tau_syn_I',  'tau_syn_i',  ms))

conductance_based_variable_translations = build_translations(
                ('v', 'v', mV),
                ('gsyn_exc', 'ge', uS),
                ('gsyn_inh', 'gi', uS))

current_based_variable_translations = build_translations(
                ('v', 'v', mV),
                ('isyn_exc', 'ie', nA),
                ('isyn_inh', 'ii', nA))


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__
    eqs = leaky_iaf + current_based_alpha_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(current_based_synapse_translations)
    state_variable_translations = current_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian_model = ThresholdNeuronGroup


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__
    eqs = leaky_iaf + current_based_exponential_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(current_based_synapse_translations)
    state_variable_translations = current_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ie', 'inhibitory': 'ii'}
    brian_model = ThresholdNeuronGroup


class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__
    eqs = leaky_iaf + conductance_based_alpha_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = conductance_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian_model = ThresholdNeuronGroup


class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__
    eqs = leaky_iaf + conductance_based_exponential_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = conductance_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = ThresholdNeuronGroup


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__
    eqs = adexp_iaf + conductance_based_exponential_synapses
    translations = deepcopy(adexp_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v', mV),
                ('w', 'w', nA),
                ('gsyn_exc', 'ge', uS),
                ('gsyn_inh', 'gi', uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = AdaptiveNeuronGroup


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__
    eqs = adexp_iaf + conductance_based_alpha_synapses
    translations = deepcopy(adexp_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v', mV),
                ('w', 'w', nA),
                ('gsyn_exc', 'ge', uS),
                ('gsyn_inh', 'gi', uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = AdaptiveNeuronGroup


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    eqs = adapt_iaf + conductance_based_alpha_synapses
    translations = deepcopy(adapt_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v', mV),
                ('g_s', 'g_s', nS),  # should be uS - needs changed for all back-ends
                ('g_r', 'g_r', nS),
                ('gsyn_exc', 'ge', uS),
                ('gsyn_inh', 'gi', uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = AdaptiveNeuronGroup2


class HH_cond_exp(cells.HH_cond_exp):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'gbar_Na',    uS),
        ('gbar_K',     'gbar_K',     uS),
        ('g_leak',     'g_leak',     uS),
        ('cm',         'c_m',        nF),
        ('v_offset',   'v_offset',   mV),
        ('e_rev_Na',   'e_rev_Na',   mV),
        ('e_rev_K',    'e_rev_K',    mV),
        ('e_rev_leak', 'e_rev_leak', mV),
        ('e_rev_E',    'e_rev_e',    mV),
        ('e_rev_I',    'e_rev_i',    mV),
        ('tau_syn_E',  'tau_syn_e',  ms),
        ('tau_syn_I',  'tau_syn_i',  ms),
        ('i_offset',   'i_offset',   nA),
    )
    eqs= brian.Equations('''
        dv/dt = (g_leak*(e_rev_leak-v) - gbar_Na*(m*m*m)*h*(v-e_rev_Na) - gbar_K*(n*n*n*n)*(v-e_rev_K) + i_syn + i_offset + i_inj)/c_m : mV
        dm/dt  = (alpham*(1-m)-betam*m) : 1
        dn/dt  = (alphan*(1-n)-betan*n) : 1
        dh/dt  = (alphah*(1-h)-betah*h) : 1
        alpham = 0.32*(mV**-1)*(13*mV-v+v_offset)/(exp((13*mV-v+v_offset)/(4*mV))-1.)/ms  : Hz
        betam  = 0.28*(mV**-1)*(v-v_offset-40*mV)/(exp((v-v_offset-40*mV)/(5*mV))-1)/ms   : Hz
        alphah = 0.128*exp((17*mV-v+v_offset)/(18*mV))/ms                                 : Hz
        betah  = 4./(1+exp((40*mV-v+v_offset)/(5*mV)))/ms                                 : Hz
        alphan = 0.032*(mV**-1)*(15*mV-v+v_offset)/(exp((15*mV-v+v_offset)/(5*mV))-1.)/ms : Hz
        betan  = .5*exp((10*mV-v+v_offset)/(40*mV))/ms                                    : Hz
        e_rev_Na               : mV
        e_rev_K                : mV
        e_rev_leak             : mV
        gbar_Na                : uS
        gbar_K                 : uS
        g_leak                 : uS
        v_offset               : mV
        c_m                    : nF
        i_offset               : nA
        i_inj                  : nA
    ''') + conductance_based_exponential_synapses
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    state_variable_translations = conductance_based_variable_translations
    brian_model = BiophysicalNeuronGroup


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',          'a',          1/ms),
        ('b',          'b',          1/ms),
        ('c',          'v_reset',    mV),
        ('d',          'd',          mV/ms),
        ('i_offset',   'i_offset',   nA)
    )
    eqs = brian.Equations('''
        dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u + (i_offset + i_inj)/pF : mV
        du/dt = a*(b*v-u)                                : mV/ms
        a                                                : 1/ms
        b                                                : 1/ms
        v_reset                                          : mV
        d                                                : mV/ms
        i_offset                                         : nA
        i_inj                                            : nA
        ''')
    post_synaptic_variables  = {'excitatory': 'v', 'inhibitory': 'v'}
    state_variable_translations =  build_translations(
                ('v', 'v', mV),
                ('u', 'u', mV/ms))
    brian_model = IzhikevichNeuronGroup
    

class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('rate',     'firing_rate',        Hz),
        ('start',    'start',       ms),
        ('duration', 'duration',    ms),
    )
    eqs = None
    brian_model = PoissonGroup
    

class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spike_times', ms),
    )
    eqs = None
    brian_model = SpikeGeneratorGroup

