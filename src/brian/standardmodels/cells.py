"""
Standard cells for the brian module

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from pyNN.standardmodels import cells, build_translations, ModelNotAvailable
import brian
from pyNN.brian.cells import PoissonGroupWithDelays, ThresholdNeuronGroup, \
                             AdaptiveNeuronGroup, BiophysicalNeuronGroup, \
                             MultipleSpikeGeneratorGroupWithDelays, \
                             IzhikevichNeuronGroup

from brian import mV, ms, nF, nA, uS, Hz


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('cm',         'c_m',        nF),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset',   nA),
    )
    eqs= brian.Equations('''
        dv/dt  = (ge + gi + i_offset + i_inj)/c_m + (v_rest-v)/tau_m : mV
        dge/dt = (2.7182818284590451*ye-ge)/tau_syn_E : nA
        dye/dt = -ye/tau_syn_E                        : nA
        dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_I : nA
        dyi/dt = -yi/tau_syn_I                        : nA
        c_m                                   : nF
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        v_rest                                : mV
        i_offset                              : nA
        i_inj                                 : nA
        '''
        )
    synapses  = {'excitatory' : 'ye', 'inhibitory' : 'yi'}
    brian_model = ThresholdNeuronGroup


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('cm',         'c_m',        nF),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset',   nA),
    )
    eqs= brian.Equations('''
        dv/dt  = (ie + ii + i_offset + i_inj)/c_m + (v_rest-v)/tau_m : mV
        die/dt = -ie/tau_syn_E                : nA
        dii/dt = -ii/tau_syn_I                : nA
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        c_m                                   : nF
        v_rest                                : mV
        i_offset                              : nA
        i_inj                                 : nA
        '''
        )
    synapses  = {'excitatory': 'ie', 'inhibitory': 'ii'}
    brian_model = ThresholdNeuronGroup

class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('cm',         'c_m',        nF),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset',   nA),
        ('e_rev_E',    'e_rev_E',    mV),
        ('e_rev_I',    'e_rev_I',    mV),
    )
    eqs= brian.Equations('''
        dv/dt  = (v_rest-v)/tau_m + (ge*(e_rev_E-v) + gi*(e_rev_I-v) + i_offset + i_inj)/c_m : mV
        dge/dt = (2.7182818284590451*ye-ge)/tau_syn_E  : uS
        dye/dt = -ye/tau_syn_E                         : uS
        dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_I  : uS
        dyi/dt = -yi/tau_syn_I                         : uS
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        v_rest                                : mV
        e_rev_E                               : mV
        e_rev_I                               : mV
        c_m                                   : nF
        i_offset                              : nA
        i_inj                                 : nA
        '''
        )
    synapses  = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian_model = ThresholdNeuronGroup



class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('cm',         'c_m',        nF),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset',   nA),
        ('e_rev_E',    'e_rev_E',    mV),
        ('e_rev_I',    'e_rev_I',    mV),
    )
    eqs= brian.Equations('''
        dv/dt  = (v_rest-v)/tau_m + (ge*(e_rev_E-v) + gi*(e_rev_I-v) + i_offset + i_inj)/c_m : mV
        dge/dt = -ge/tau_syn_E : uS
        dgi/dt = -gi/tau_syn_I : uS
        tau_syn_E              : ms
        tau_syn_I              : ms
        tau_m                  : ms
        c_m                    : nF
        v_rest                 : mV
        e_rev_E                : mV
        e_rev_I                : mV
        i_offset               : nA
        i_inj                  : nA
        '''
        )
    synapses  = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = ThresholdNeuronGroup



class IF_facets_hardware1(ModelNotAvailable):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    pass


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('rate',     'rate',        Hz),
        ('start',    'start',       ms),
        ('duration', 'duration',    ms),
    )
    eqs = None
    brian_model = PoissonGroupWithDelays


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spiketimes', ms),
    )
    eqs = None
    brian_model = MultipleSpikeGeneratorGroupWithDelays


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'c_m',        nF),
        ('v_spike',    'v_spike'),
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('i_offset',   'i_offset',   nA),
        ('a',          'a',          nA),
        ('b',          'b',          nA),
        ('delta_T',    'delta_T',    mV),
        ('tau_w',      'tau_w',      ms),
        ('v_thresh',   'v_thresh',   mV),
        ('e_rev_E',    'e_rev_E',    mV),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('e_rev_I',    'e_rev_I',    mV),
        ('tau_syn_I',  'tau_syn_I',  ms),
    )
    eqs= brian.Equations('''
        dv/dt  = ((v_rest-v) + delta_T*exp((v-v_thresh)/delta_T))/tau_m + (ge*(e_rev_E-v) + gi*(e_rev_I-v) + i_offset + i_inj - w)/c_m : mV
        dge/dt = (2.7182818284590451*ye-ge)/tau_syn_E  : uS
        dye/dt = -ye/tau_syn_E                         : uS
        dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_I  : uS
        dyi/dt = -yi/tau_syn_I                         : uS
        dw/dt  = (a*(v-v_rest) - w)/tau_w : nA
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        v_rest                                : mV
        e_rev_E                               : mV
        e_rev_I                               : mV
        c_m                                   : nF
        i_offset                              : nA
        i_inj                                 : nA
        delta_T                               : mV
        a                                     : uS
        b                                     : nA
        tau_w                                 : ms
        v_thresh                              : mV
        v_spike                               : mV
        '''
        )
    synapses  = {'excitatory': 'ye', 'inhibitory': 'yi'}
    brian_model = AdaptiveNeuronGroup


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'c_m',        nF),
        ('v_spike',    'v_spike'),
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset'),
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac'),
        ('i_offset',   'i_offset',   nA),
        ('a',          'a',          nA),
        ('b',          'b',          nA),
        ('delta_T',    'delta_T',    mV),
        ('tau_w',      'tau_w',      ms),
        ('v_thresh',   'v_thresh',   mV),
        ('e_rev_E',    'e_rev_E',    mV),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('e_rev_I',    'e_rev_I',    mV),
        ('tau_syn_I',  'tau_syn_I',  ms),
    )
    eqs= brian.Equations('''
        dv/dt  = ((v_rest-v) + delta_T*exp((v - v_thresh)/delta_T))/tau_m + (ge*(e_rev_E-v) + gi*(e_rev_I-v) + i_offset + i_inj - w)/c_m : mV
        dge/dt = -ge/tau_syn_E                : uS
        dgi/dt = -gi/tau_syn_I                : uS
        dw/dt  = (a*(v-v_rest) - w)/tau_w  : nA
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        v_rest                                : mV
        e_rev_E                               : mV
        e_rev_I                               : mV
        c_m                                   : nF
        i_offset                              : nA
        i_inj                                 : nA
        delta_T                               : mV
        a                                     : uS
        b                                     : nA
        tau_w                                 : ms
        v_thresh                              : mV
        v_spike                               : mV
        '''
        )
    synapses  = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = AdaptiveNeuronGroup


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
        ('e_rev_E',    'e_rev_E',    mV),
        ('e_rev_I',    'e_rev_I',    mV),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('i_offset',   'i_offset',   nA),
    )
    eqs= brian.Equations('''
        dv/dt = (g_leak*(e_rev_leak-v)+ge*(e_rev_E-v)+gi*(e_rev_I-v)-gbar_Na*(m*m*m)*h*(v-e_rev_Na)-gbar_K*(n*n*n*n)*(v-e_rev_K) + i_offset + i_inj)/c_m : mV
        dm/dt  = (alpham*(1-m)-betam*m) : 1
        dn/dt  = (alphan*(1-n)-betan*n) : 1
        dh/dt  = (alphah*(1-h)-betah*h) : 1
        dge/dt = -ge/tau_syn_E : uS
        dgi/dt = -gi/tau_syn_I : uS
        alpham = 0.32*(mV**-1)*(13*mV-v+v_offset)/(exp((13*mV-v+v_offset)/(4*mV))-1.)/ms  : Hz
        betam  = 0.28*(mV**-1)*(v-v_offset-40*mV)/(exp((v-v_offset-40*mV)/(5*mV))-1)/ms   : Hz
        alphah = 0.128*exp((17*mV-v+v_offset)/(18*mV))/ms                                 : Hz
        betah  = 4./(1+exp((40*mV-v+v_offset)/(5*mV)))/ms                                 : Hz
        alphan = 0.032*(mV**-1)*(15*mV-v+v_offset)/(exp((15*mV-v+v_offset)/(5*mV))-1.)/ms : Hz
        betan  = .5*exp((10*mV-v+v_offset)/(40*mV))/ms                                    : Hz
        tau_syn_E              : ms
        tau_syn_I              : ms
        e_rev_E                : mV
        e_rev_I                : mV
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
    ''')
    synapses  = {'excitatory': 'ge', 'inhibitory': 'gi'}
    brian_model = BiophysicalNeuronGroup


class SpikeSourceInhGamma(ModelNotAvailable):
    pass


class IF_cond_exp_gsfa_grr(ModelNotAvailable):
    pass


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',    'a', 1/ms),
        ('b',    'b', 1/ms),
        ('v_reset', 'v_reset'),
        ('d',    'd', mV/ms),
        ('tau_refrac', 'tau_refrac')
    )
    eqs = brian.Equations('''
        dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms - u + (ie + ii) * (mV/ms) / nA : mV
        du/dt = a*(b*v-u)                                : mV/ms
        die/dt = -ie/(1*ms)                              : nA
        dii/dt = -ii/(1*ms)                              : nA
        a                                                : 1/ms
        b                                                : 1/ms
        v_reset                                          : mV
        d                                                : mV/ms
        ''')
    synapses  = {'excitatory': 'ie', 'inhibitory': 'ii'}
    brian_model = IzhikevichNeuronGroup
