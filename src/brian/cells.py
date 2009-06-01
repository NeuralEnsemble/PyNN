"""
Standard cells for the brian module

$Id$
"""

from pyNN import common, cells
#import brian_no_units_no_warnings
from brian.library.synapses import *
import brian
from brian import mV, ms, nF, nA, uS, second, Hz


class IF_curr_alpha(cells.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    translations = common.build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset',    mV),
        ('cm',         'c_m',         nF), 
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac', ms),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh',   ms),
        ('i_offset',   'i_offset',   nA), 
        ('v_init',     'v_init',     ms),
    )
    eqs= brian.Equations('''
        dv/dt  = (ge + gi + i_offset + i_inj)/c_m + (v_rest-v)/tau_m : mV
        dge/dt = (2.7182818284590451*ye-ge)/tau_syn_E : nA
        dye/dt = -ye/tau_syn_E                        : nA
        dgi/dt = (2.7182818284590451*yi-gi)/tau_syn_I : nA
        dyi/dt = -yi/tau_syn_I                        : nA
        c_m                                    : nF
        tau_syn_E                             : ms
        tau_syn_I                             : ms
        tau_m                                 : ms
        v_rest                                : mV
        i_offset                              : nA
        i_inj                                 : nA
        '''
        )
    synapses = {'excitatory' : 'ye', 'inhibitory' : 'yi'}


class IF_curr_exp(cells.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset',    mV),
        ('cm',         'c_m',        nF), 
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac', ms),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh',   ms),
        ('i_offset',   'i_offset',   nA), 
        ('v_init',     'v_init',     mV),
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
    
    synapses = {'excitatory': 'ie', 'inhibitory': 'ii'}
    

class IF_cond_alpha(cells.IF_cond_alpha):
    translations = common.build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset',    mV),
        ('cm',         'c_m',        nF), 
        ('tau_m',      'tau_m',      mV),
        ('tau_refrac', 'tau_refrac', ms),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh',   mV),
        ('i_offset',   'i_offset',   nA), 
        ('e_rev_E',    'e_rev_E',    mV),
        ('e_rev_I',    'e_rev_I',    mV),
        ('v_init',     'v_init',     mV),
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
    synapses = {'excitatory': 'ye', 'inhibitory': 'yi'}

class IF_cond_exp(cells.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and 
    exponentially-decaying post-synaptic conductance."""
    translations = common.build_translations(
        ('v_rest',     'v_rest',     mV),
        ('v_reset',    'v_reset',    mV),
        ('cm',         'cm',         nF), 
        ('tau_m',      'tau_m',      ms),
        ('tau_refrac', 'tau_refrac', ms),
        ('tau_syn_E',  'tau_syn_E',  ms),
        ('tau_syn_I',  'tau_syn_I',  ms),
        ('v_thresh',   'v_thresh',   mV),
        ('i_offset',   'i_offset',   nA), 
        ('e_rev_E',    'e_rev_E',    mV),
        ('e_rev_I',    'e_rev_I',    mV),
        ('v_init',     'v_init',     mV),
    )
    eqs= brian.Equations('''
        dv/dt  = (v_rest-v)/tau_m + (ge*(e_rev_E-v) + gi*(e_rev_I-v) + i_offset + i_inj)/cm : mV
        dge/dt = -ge/tau_syn_E : uS
        dgi/dt = -gi/tau_syn_I : uS
        tau_syn_E              : ms
        tau_syn_I              : ms
        tau_m                  : ms
        cm                     : nF
        v_rest                 : mV
        e_rev_E                : mV
        e_rev_I                : mV
        i_offset               : nA
        i_inj                  : nA
        '''
        )
    synapses = {'excitatory': 'ge', 'inhibitory': 'gi'}


class IF_facets_hardware1(common.ModelNotAvailable):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    pass


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""
    translations = common.build_translations(
        ('rate',     'rate', Hz),
        ('start',    'start', ms),
        ('duration', 'duration', ms),
    )
    
    class rates(object):
        """
        Acts as a function of time for the PoissonGroup, while storing the
        parameters for later retrieval.
        """
        def __init__(self, start, duration, rate):
            self.start = start*ms
            self.duration = duration*ms
            self.rate = rate*Hz
        def __call__(self, t):
            #print self.start
            #print self.duration
            #print self.rate
            #print t
            return (self.start <= t <= self.start + self.duration) and self.rate or 0.0*Hz
    
    def __init__(self, parameters):
        cells.SpikeSourcePoisson.__init__(self, parameters)
        start    = self.parameters['start']
        duration = self.parameters['duration']
        rate     = self.parameters['rate']
        self.fct = SpikeSourcePoisson.rates(start, duration, rate)
    
    
class SpikeSourceArray(cells.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""
    translations = common.build_translations(
        ('spike_times', 'spiketimes', ms),
    )


class EIF_cond_alpha_isfa_ista(common.ModelNotAvailable):
    pass


class HH_cond_exp(common.ModelNotAvailable):
    pass
    
#class HH_cond_exp(cells.HH_cond_exp):
#    
#    translations = common.build_translations(
#        ('gbar_Na',    'gbar_Na'),   
#        ('gbar_K',     'gbar_K'),    
#        ('g_leak',     'g_leak'),    
#        ('cm',         'cm'),  
#        ('v_offset',   'v_offset'),
#        ('e_rev_Na',   'e_rev_Na'),
#        ('e_rev_K',    'e_rev_K'), 
#        ('e_rev_leak', 'e_rev_leak'),
#        ('e_rev_E',    'e_rev_E'),
#        ('e_rev_I',    'e_rev_I'),
#        ('tau_syn_E',  'tau_syn_E'),
#        ('tau_syn_I',  'tau_syn_I'),
#        ('i_offset',   'i_offset'),
#        ('v_init',     'v_init'),
#    )
#    
#    eqs= brian.Equations('''
#        dv/dt = (g_leak*(e_rev_leak-v)+ge*(e_rev_E-v)+gi*(e_rev_I-v)-gbar_Na*(m*m*m)*h*(v-e_rev_Na)-gbar_K*(n*n*n*n)*(v-e_rev_K) + i_offset + i_inj)/cm : volt
#        dm/dt = alpham*(1-m)-betam*m : 1
#        dn/dt = alphan*(1-n)-betan*n : 1
#        dh/dt = alphah*(1-h)-betah*h : 1
#        dge/dt = -ge/tau_syn_E : siemens
#        dgi/dt = -gi/tau_syn_I : siemens
#        alpham = 0.32*(mV**-1)*(13*mV-v+VT)/(exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
#        betam = 0.28*(mV**-1)*(v-VT-40*mV)/(exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
#        alphah = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
#        betah = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
#        alphan = 0.032*(mV**-1)*(15*mV-v+VT)/(exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
#        betan = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
#        tau_syn_E              : second
#        tau_syn_I              : second
#        e_rev_E                : volt
#        e_rev_I                : volt
#        e_rev_Na               : volt
#        e_rev_K                : volt
#        e_rev_leak             : volt
#        gbar_Na                : nS
#        gbar_K                 : nS
#        g_leak                 : nS
#        v_offset               : volt
#        cm                     : nF
#        VT                     : volt
#        i_offset               : amp
#        i_inj                  : amp
#    ''')


class SpikeSourceInhGamma(common.ModelNotAvailable):
    pass


class IF_cond_exp_gsfa_grr(common.ModelNotAvailable):
    pass