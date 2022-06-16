# encoding: utf-8
"""
Standard cells for the Brian2 module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from copy import deepcopy
import brian2
from brian2 import mV, ms, nF, nA, uS, Hz, nS
from pyNN.standardmodels import cells, build_translations
from ..cells import (ThresholdNeuronGroup, SpikeGeneratorGroup, PoissonGroup,
                     BiophysicalNeuronGroup, AdaptiveNeuronGroup, AdaptiveNeuronGroup2,
                     IzhikevichNeuronGroup)
import logging

logger = logging.getLogger("PyNN")


leaky_iaf = brian2.Equations('''
                dv/dt = (v_rest-v)/tau_m + (i_syn + i_offset + i_inj)/c_m  : volt (unless refractory)
                tau_m                   : second
                c_m                     : farad
                v_rest                  : volt
                i_offset                : amp
                i_inj                   : amp
            ''')
# give v_thresh a different name
adexp_iaf = brian2.Equations('''
                dv/dt = (delta_T*gL*exp((-v_thresh + v)/delta_T) + I + gL*(v_rest - v) - w )/ c_m : volt (unless refractory)
                dw/dt = (a*(v-v_rest) - w)/tau_w  : amp
                gL    =  c_m / tau_m    : siemens
                I = i_syn + i_inj + i_offset : amp
                a                       : siemens
                tau_m                   : second
                tau_w                   : second
                c_m                     : farad
                v_rest                  : volt
                v_spike                 : volt
                v_thresh                : volt
                delta_T                 : volt
                i_offset                : amp
                i_inj                   : amp
            ''')

# g_r, g_s should be in uS for PyNN unit system consistency
adapt_iaf = brian2.Equations('''
                dv/dt = (v_rest-v)/tau_m + (-g_r*(v-E_r) - g_s*(v-E_s) + i_syn + i_offset + i_inj)/c_m  : volt (unless refractory)
                dg_s/dt = -g_s/tau_s    : siemens (unless refractory)
                dg_r/dt = -g_r/tau_r    : siemens (unless refractory)
                tau_m                   : second
                tau_s                   : second
                tau_r                   : second
                c_m                     : farad
                v_rest                  : volt
                i_offset                : amp
                i_inj                   : amp
                E_r                     : volt
                E_s                     : volt

            ''')


conductance_based_exponential_synapses = brian2.Equations('''
                dge/dt = -ge/tau_syn_e  : siemens
                dgi/dt = -gi/tau_syn_i  : siemens
                i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)  : amp
                tau_syn_e               : second
                tau_syn_i               : second
                e_rev_e                 : volt
                e_rev_i                 : volt
            ''')

conductance_based_alpha_synapses = brian2.Equations('''
                dge/dt = (2.7182818284590451*ye-ge)/tau_syn_e  : siemens
                dye/dt = -ye/tau_syn_e                         : siemens
                dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_i  : siemens
                dyi/dt = -yi/tau_syn_i                         : siemens
                i_syn = ge*(e_rev_e - v) + gi*(e_rev_i - v)    : amp
                tau_syn_e               : second
                tau_syn_i               : second
                e_rev_e                 : volt
                e_rev_i                 : volt
        ''')

current_based_exponential_synapses = brian2.Equations('''
                die/dt = -ie/tau_syn_e  : amp
                dii/dt = -ii/tau_syn_i  : amp
                i_syn = ie + ii         : amp
                tau_syn_e               : second
                tau_syn_i               : second
            ''')

current_based_alpha_synapses = brian2.Equations('''
                die/dt = (2.7182818284590451*ye-ie)/tau_syn_e : amp
                dye/dt = -ye/tau_syn_e                        : amp
                dii/dt = (2.7182818284590451*yi-ii)/tau_syn_e : amp
                dyi/dt = -yi/tau_syn_e                        : amp
                i_syn = ie + ii                               : amp
                tau_syn_e                                     : second
                tau_syn_i                                     : second
            ''')

voltage_step_synapses = brian2.Equations('''
                i_syn = 0 * amp  : amp
            ''')

leaky_iaf_translations = build_translations(
                ('v_rest',     'v_rest',     lambda **p: p["v_rest"] * mV, lambda **p: p["v_rest"] / mV),
                ('v_reset',    'v_reset',    lambda **p: p["v_reset"] * mV, lambda **p: p["v_reset"] / mV),
                ('cm',         'c_m',        lambda **p: p["cm"] * nF, lambda **p: p["c_m"] / nF),
                ('tau_m',      'tau_m',      lambda **p: p["tau_m"] * ms, lambda **p: p["tau_m"] / ms), ###p["tau_m"] * ms, p["tau_m"] /nF
                ('tau_refrac', 'tau_refrac', lambda **p: p["tau_refrac"] * ms, lambda **p: p["tau_refrac"] / ms),
                ('v_thresh',   'v_thresh',   lambda **p: p["v_thresh"] * mV, lambda **p: p["v_thresh"] / mV),
                ('i_offset',   'i_offset',   lambda **p: p["i_offset"] * nA, lambda **p: p["i_offset"] / nA))

adexp_iaf_translations = build_translations(

                ('v_rest',     'v_rest',     lambda **p: p["v_rest"] * mV, lambda **p: p["v_rest"] / mV),
                ('v_reset',    'v_reset',   lambda **p: p["v_reset"] * mV, lambda **p: p["v_reset"] / mV),
                ('cm',         'c_m',        lambda **p: p["cm"] * nF, lambda **p: p["c_m"] / nF),
                ('tau_m',      'tau_m',     lambda **p: p["tau_m"] * ms, lambda **p: p["tau_m"] / ms),
                ('tau_refrac', 'tau_refrac', lambda **p: p["tau_refrac"] * ms, lambda **p: p["tau_refrac"] / ms),
                ('v_thresh',   'v_thresh',  lambda **p: p["v_thresh"] * mV, lambda **p: p["v_thresh"] / mV),
                ('i_offset',   'i_offset',  lambda **p: p["i_offset"] * nA, lambda **p: p["i_offset"] / nA),
                ('a',          'a',          lambda **p: p["a"] * nS, lambda **p: p["a"] / nS),
                ('b',          'b',          lambda **p: p["b"] * nA, lambda **p: p["b"] / nA),
                ('delta_T',    'delta_T',   lambda **p: p["delta_T"] * mV, lambda **p: p["delta_T"] / mV),
                ('tau_w',      'tau_w',     lambda **p: p["tau_w"] * ms, lambda **p: p["tau_w"] / ms),
                ('v_spike',    'v_spike',    lambda **p: p["v_spike"] * mV, lambda **p: p["v_spike"] / mV))

adapt_iaf_translations = build_translations(
                ('v_rest',     'v_rest',     lambda **p: p["v_rest"] * mV, lambda **p: p["v_rest"] / mV),
                ('v_reset',    'v_reset',    lambda **p: p["v_reset"] * mV, lambda **p: p["v_reset"] / mV),
                ('cm',         'c_m',        lambda **p: p["cm"] * nF, lambda **p: p["c_m"] / nF),
                ('tau_m',      'tau_m',      lambda **p: p["tau_m"] * ms, lambda **p: p["tau_m"] / ms),
                ('tau_refrac', 'tau_refrac', lambda **p: p["tau_refrac"] * ms, lambda **p: p["tau_refrac"] / ms),
                ('v_thresh',   'v_thresh',   lambda **p: p["v_thresh"] * mV, lambda **p: p["v_thresh"] / mV),
                ('i_offset',   'i_offset',   lambda **p: p["i_offset"] * nA, lambda **p: p["i_offset"] / nA),
                ('tau_sfa',    'tau_s',      lambda **p: p["tau_sfa"] * ms, lambda **p: p["tau_s"] / ms),
                ('e_rev_sfa',  'E_s',        lambda **p: p["e_rev_sfa"] * mV, lambda **p: p["E_s"] / mV),
                ('q_sfa',      'q_s',        lambda **p: p["q_sfa"] * nS, lambda **p: p["q_s"] / nS),   # should we uS for consistency of PyNN unit system?
                ('tau_rr',     'tau_r',      lambda **p: p["tau_rr"] * ms, lambda **p: p["tau_r"] / ms),
                ('e_rev_rr',   'E_r',        lambda **p: p["e_rev_rr"] * mV, lambda **p: p["E_r"] / mV),
                ('q_rr',       'q_r',        lambda **p: p["q_rr"] * nS, lambda **p: p["q_r"] / nS))

conductance_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  lambda **p: p["tau_syn_E"] * ms, lambda **p: p["tau_syn_e"] / ms),
                ('tau_syn_I',  'tau_syn_i',  lambda **p: p["tau_syn_I"] * ms, lambda **p: p["tau_syn_i"] / ms),
                ('e_rev_E',    'e_rev_e',    lambda **p: p["e_rev_E"] * mV, lambda **p: p["e_rev_e"] / mV),
                ('e_rev_I',    'e_rev_i',    lambda **p: p["e_rev_I"] * mV, lambda **p: p["e_rev_i"] / mV))

current_based_synapse_translations = build_translations(
                ('tau_syn_E',  'tau_syn_e',  lambda **p: p["tau_syn_E"] * ms, lambda **p: p["tau_syn_e"] / ms),
                ('tau_syn_I',  'tau_syn_i',  lambda **p: p["tau_syn_I"] * ms, lambda **p: p["tau_syn_i"] / ms))

conductance_based_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('gsyn_exc', 'ge', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS))
current_based_variable_translations = build_translations(
                ('v',         'v',         lambda p: p * mV, lambda p: p/ mV), #### change p by p["v"]
                ('isyn_exc', 'ie',         lambda p: p * nA, lambda p: p/ nA),
                ('isyn_inh', 'ii',         lambda p: p * nA, lambda p: p/ nA))


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__
    eqs = leaky_iaf + current_based_alpha_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(current_based_synapse_translations)
    state_variable_translations = current_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian2_model = ThresholdNeuronGroup


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__
    eqs = leaky_iaf + current_based_exponential_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(current_based_synapse_translations)
    state_variable_translations = current_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ie', 'inhibitory': 'ii'}
    brian2_model = ThresholdNeuronGroup


class IF_curr_delta(cells.IF_curr_delta):
    __doc__ = cells.IF_curr_delta.__doc__
    eqs = leaky_iaf + voltage_step_synapses
    translations = deepcopy(leaky_iaf_translations)
    state_variable_translations = build_translations(
        ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
    )
    post_synaptic_variables = {'excitatory': 'v', 'inhibitory': 'v'}
    brian2_model = ThresholdNeuronGroup


class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__
    eqs = leaky_iaf + conductance_based_alpha_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = conductance_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian2_model = ThresholdNeuronGroup


class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__
    eqs = leaky_iaf + conductance_based_exponential_synapses
    translations = deepcopy(leaky_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = conductance_based_variable_translations
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = ThresholdNeuronGroup


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__
    eqs = adexp_iaf + conductance_based_exponential_synapses
    translations = deepcopy(adexp_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v',lambda p: p * mV, lambda p: p/ mV),
                ('w', 'w', lambda p: p * nA, lambda p: p/ nA),
                ('gsyn_exc', 'ge',lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS))

    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = AdaptiveNeuronGroup


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__
    eqs = adexp_iaf + conductance_based_alpha_synapses
    translations = deepcopy(adexp_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('w', 'w', lambda p: p * nA, lambda p: p/ nA),
                ('gsyn_exc', 'ge', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = AdaptiveNeuronGroup


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    eqs = adapt_iaf + conductance_based_alpha_synapses
    translations = deepcopy(adapt_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('g_s', 'g_s', lambda p: p * uS, lambda p: p/ uS), # should be uS - needs changed for all back-ends
                ('g_r', 'g_r', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_exc', 'ge', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = AdaptiveNeuronGroup2


class HH_cond_exp(cells.HH_cond_exp):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'gbar_Na',    lambda **p: p["gbar_Na"] * uS, lambda **p: p["gbar_Na"] / uS),
        ('gbar_K',     'gbar_K',     lambda **p: p["gbar_K"] * uS, lambda **p: p["gbar_K"] / uS),
        ('g_leak',     'g_leak',     lambda **p: p["g_leak"] * uS, lambda **p: p["g_leak"] / uS),
        ('cm',         'c_m',        lambda **p: p["cm"] * nF, lambda **p: p["c_m"] / nF),
        ('v_offset',   'v_offset',   lambda **p: p["v_offset"] * mV, lambda **p: p["v_offset"] / mV),
        ('e_rev_Na',   'e_rev_Na',   lambda **p: p["e_rev_Na"] * mV, lambda **p: p["e_rev_Na"] / mV),
        ('e_rev_K',    'e_rev_K',    lambda **p: p["e_rev_K"] * mV, lambda **p: p["e_rev_K"] / mV),
        ('e_rev_leak', 'e_rev_leak', lambda **p: p["e_rev_leak"] * mV, lambda **p: p["e_rev_leak"] / mV),
        ('e_rev_E',    'e_rev_e',    lambda **p: p["e_rev_E"] * mV, lambda **p: p["e_rev_e"] / mV),
        ('e_rev_I',    'e_rev_i',    lambda **p: p["e_rev_I"] * mV, lambda **p: p["e_rev_i"] / mV),
        ('tau_syn_E',  'tau_syn_e',  lambda **p: p["tau_syn_E"] * ms, lambda **p: p["tau_syn_e"] / ms),
        ('tau_syn_I',  'tau_syn_i',  lambda **p: p["tau_syn_I"] * ms, lambda **p: p["tau_syn_i"] / ms),
        ('i_offset',   'i_offset',   lambda **p: p["i_offset"] * nA, lambda **p: p["i_offset"] / nA))
    eqs = brian2.Equations('''
        dv/dt =  (g_leak*(e_rev_leak-v) - gbar_Na*(m*m*m)*h*(v-e_rev_Na) - gbar_K*(n*n*n*n)*(v-e_rev_K) + i_syn + i_offset + i_inj)/c_m : volt
        dm/dt  = (alpham*(1-m)-betam*m) : 1
        dn/dt  = (alphan*(1-n)-betan*n) : 1
        dh/dt  = (alphah*(1-h)-betah*h) : 1
        alpham = (0.32/mV)*(13*mV-v+v_offset)/(exp((13*mV-v+v_offset)/(4*mV))-1.)/ms  : Hz
        betam  = (0.28/mV)*(v-v_offset-40*mV)/(exp((v-v_offset-40*mV)/(5*mV))-1)/ms   : Hz
        alphah = 0.128*exp((17*mV-v+v_offset)/(18*mV))/ms                                 : Hz
        betah  = 4./(1+exp((40*mV-v+v_offset)/(5*mV)))/ms                                 : Hz
        alphan = (0.032/mV)*(15*mV-v+v_offset)/(exp((15*mV-v+v_offset)/(5*mV))-1.)/ms : Hz
        betan  = .5*exp((10*mV-v+v_offset)/(40*mV))/ms                                    : Hz
        e_rev_Na               : volt
        e_rev_K                : volt
        e_rev_leak             : volt
        gbar_Na                : siemens
        gbar_K                 : siemens
        g_leak                 : siemens
        v_offset               : volt
        c_m                    : farad
        i_offset               : amp
        i_inj                  : amp
    ''') + conductance_based_exponential_synapses
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh', 'm','n','h']
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    state_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('gsyn_exc', 'ge', lambda p: p * uS, lambda p: p/ uS),
                ('gsyn_inh', 'gi', lambda p: p * uS, lambda p: p/ uS),
                ('h', 'h', lambda p: p *1, lambda p: p*1),
                ('m', 'm', lambda p: p *1, lambda p: p*1),
                ('n', 'n', lambda p: p*1 , lambda p: p*1))
    brian2_model = BiophysicalNeuronGroup


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',          'a',          lambda **p: p["a"] *(1/ms) , lambda **p: p["a"] / (1/ms)),
        ('b',          'b',          lambda **p: p["b"] *(1/ms) , lambda **p: p["b"] / (1/ms)),
        ('c',          'v_reset',    lambda **p: p["c"] * mV, lambda **p: p["v_reset"] / mV),
        ('d',          'd',          lambda **p: p["d"] *(mV/ms) , lambda **p: p["d"] / (mV/ms)),
        ('i_offset',   'i_offset',   lambda **p: p["i_offset"] * nA, lambda **p: p["i_offset"] / nA))
    ### dv/dt = (0.04/ms/mV)*v*v ->>>> (0.04/ms/mV)*v**2
    eqs = brian2.Equations('''
        dv/dt = (0.04/ms/mV)*v*v + (5/ms)*v + 140*mV/ms - u + (i_offset + i_inj)/pF : volt (unless refractory)
        du/dt = a*(b*v-u)                                : volt/second (unless refractory)
        a                                                : 1/second
        b                                                : 1/second
        v_reset                                          : volt
        d                                                : volt/second
        i_offset                                         : amp
        i_inj                                            : amp
        ''')
    post_synaptic_variables = {'excitatory': 'v', 'inhibitory': 'v'}
    state_variable_translations = build_translations(
                ('v', 'v', lambda p: p * mV, lambda p: p/ mV),
                ('u', 'u', lambda p: p * (mV/ms), lambda p: p/ (mV/ms)))
    brian2_model = IzhikevichNeuronGroup


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('rate', 'firing_rate', lambda **p: p["rate"] * Hz, lambda **p: p["firing_rate"] / Hz),
        ('start', 'start_time', lambda **p: p["start"] * ms, lambda **p: p["start_time"] / ms),
        ('duration', 'duration', lambda **p: p["duration"] * ms, lambda **p: p["duration"] / ms),
    )
    eqs = None
    brian2_model = PoissonGroup


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spike_time_sequences', ms),
    )
    eqs = None
    brian2_model = SpikeGeneratorGroup
