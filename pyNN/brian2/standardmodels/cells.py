# encoding: utf-8
"""
Standard cells for the Brian2 module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa

from copy import deepcopy
import logging
import brian2
from brian2 import mV, ms, nF, nA, uS, Hz, nS
from ...standardmodels import cells, build_translations
from ..cells import (ThresholdNeuronGroup, SpikeGeneratorGroup, PoissonGroup,
                     BiophysicalNeuronGroup, AdaptiveNeuronGroup, AdaptiveNeuronGroup2,
                     IzhikevichNeuronGroup)
from .receptors import (conductance_based_alpha_synapses,
                        conductance_based_exponential_synapses,
                        current_based_alpha_synapses,
                        current_based_exponential_synapses,
                        voltage_step_synapses,
                        conductance_based_synapse_translations,
                        current_based_synapse_translations,
                        conductance_based_variable_translations,
                        current_based_variable_translations)

logger = logging.getLogger("PyNN")


leaky_iaf = brian2.Equations('''
    dv/dt = (v_rest-v)/tau_m + (i_syn + i_offset + i_inj)/c_m  : volt (unless refractory)
    tau_m                   : second
    c_m                     : farad
    v_rest                  : volt
    i_offset                : amp
    i_inj                   : amp
''')

adexp_iaf = brian2.Equations('''
    dv/dt = (delta_T*gL*exp((-v_thresh + v)/delta_T) + I + gL*(v_rest - v) - w )/ c_m : volt (unless refractory)  # noqa: E501
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
    dv/dt = (v_rest-v)/tau_m + (-g_r*(v-E_r) - g_s*(v-E_s) + i_syn + i_offset + i_inj)/c_m  : volt (unless refractory)  # noqa: E501
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
    ('a',          'a',          nS),
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
    ('q_sfa',      'q_s',        nS),
    # should we uS for consistency of PyNN unit system?
    ('tau_rr',     'tau_r',      ms),
    ('e_rev_rr',   'E_r',        mV),
    ('q_rr',       'q_r',        nS))


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
        ('v', 'v', mV),
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
        ('v', 'v', mV),
        ('w', 'w', nA),
        ('gsyn_exc', 'ge', uS),
        ('gsyn_inh', 'gi', uS))

    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = AdaptiveNeuronGroup


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
    brian2_model = AdaptiveNeuronGroup


class LIF(cells.LIF):
    eqs = leaky_iaf
    translations = deepcopy(leaky_iaf_translations)
    state_variable_translations = build_translations(
        ('v', 'v', mV),
    )
    brian2_model = ThresholdNeuronGroup


class AdExp(cells.AdExp):
    eqs = adexp_iaf
    translations = deepcopy(adexp_iaf_translations)
    state_variable_translations = build_translations(
        ('v', 'v', mV),
        ('w', 'w', nA)
    )
    brian2_model = AdaptiveNeuronGroup


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    eqs = adapt_iaf + conductance_based_alpha_synapses
    translations = deepcopy(adapt_iaf_translations)
    translations.update(conductance_based_synapse_translations)
    state_variable_translations = build_translations(
        ('v', 'v', mV),
        ('g_s', 'g_s', uS),
        ('g_r', 'g_r', uS),
        ('gsyn_exc', 'ge', uS),
        ('gsyn_inh', 'gi', uS))
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian2_model = AdaptiveNeuronGroup2


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
        ('i_offset',   'i_offset',   nA))
    eqs = brian2.Equations('''
        dv/dt =  (g_leak*(e_rev_leak-v) - gbar_Na*(m*m*m)*h*(v-e_rev_Na) - gbar_K*(n*n*n*n)*(v-e_rev_K) + i_syn + i_offset + i_inj)/c_m : volt  # noqa: E501
        dm/dt  = (alpham*(1-m)-betam*m) : 1
        dn/dt  = (alphan*(1-n)-betan*n) : 1
        dh/dt  = (alphah*(1-h)-betah*h) : 1
        alpham = (0.32/mV)*(13*mV-v+v_offset)/(exp((13*mV-v+v_offset)/(4*mV))-1.)/ms  : Hz
        betam  = (0.28/mV)*(v-v_offset-40*mV)/(exp((v-v_offset-40*mV)/(5*mV))-1)/ms   : Hz
        alphah = 0.128*exp((17*mV-v+v_offset)/(18*mV))/ms                             : Hz
        betah  = 4./(1+exp((40*mV-v+v_offset)/(5*mV)))/ms                             : Hz
        alphan = (0.032/mV)*(15*mV-v+v_offset)/(exp((15*mV-v+v_offset)/(5*mV))-1.)/ms : Hz
        betan  = .5*exp((10*mV-v+v_offset)/(40*mV))/ms                                : Hz
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
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh', 'm', 'n', 'h']
    post_synaptic_variables = {'excitatory': 'ge', 'inhibitory': 'gi'}
    state_variable_translations = build_translations(
        ('v', 'v', mV),
        ('gsyn_exc', 'ge', uS),
        ('gsyn_inh', 'gi', uS),
        ('h', 'h'),
        ('m', 'm'),
        ('n', 'n'))
    brian2_model = BiophysicalNeuronGroup


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',        'a',          1 / ms),
        ('b',        'b',          1 / ms),
        ('c',        'v_reset',    mV),
        ('d',        'd',          mV / ms),
        ('i_offset', 'i_offset',   nA))
    eqs = brian2.Equations('''
        dv/dt = (0.04/ms/mV)*v*v + (5/ms)*v + 140*mV/ms - u + (i_offset + i_inj)/pF : volt (unless refractory)  # noqa: E501
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
        ('v', 'v', mV),
        ('u', 'u', mV / ms))
    brian2_model = IzhikevichNeuronGroup


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('rate', 'firing_rate', Hz),
        ('start', 'start_time', ms),
        ('duration', 'duration', ms),
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


class PointNeuron(cells.PointNeuron):

    def __init__(self, neuron, **post_synaptic_receptors):
        super(PointNeuron, self).__init__(neuron, **post_synaptic_receptors)
        self.eqs = neuron.eqs
        self.translations = deepcopy(neuron.translations)
        self.state_variable_translations = neuron.state_variable_translations
        self.post_synaptic_variables = {}
        synaptic_current_equation = "i_syn ="
        for psr_label, psr in post_synaptic_receptors.items():
            self.eqs += psr.eqs(psr_label)
            self.translations.update(psr.translations(psr_label))
            self.state_variable_translations.update(psr.state_variable_translations(psr_label))
            self.post_synaptic_variables.update({psr_label: psr.post_synaptic_variable(psr_label)})
            synaptic_current_equation += f" {psr.synaptic_current(psr_label)} +"
        synaptic_current_equation = synaptic_current_equation.strip("+")
        synaptic_current_equation += "  : amp"
        self.eqs += brian2.Equations(synaptic_current_equation)
        self.brian2_model = neuron.brian2_model

    def get_native_names(self, *names):
        neuron_names = self.neuron.get_native_names(*[name for name in names if "." not in name])
        for name in names:
            if "." in name:
                psr_name, param_name = name.split(".")
                index = self.receptor_types.index(psr_name)
                tr_name = self.post_synaptic_receptors[psr_name].get_native_names(param_name)
                neuron_names.append("{}[{}]".format(tr_name, index))
        return neuron_names

    @property
    def native_parameters(self):
        translated_parameters = self.neuron.native_parameters
        for name, psr in self.post_synaptic_receptors.items():
            translated_parameters.add_child(name, psr.native_parameters(name))
        return translated_parameters
