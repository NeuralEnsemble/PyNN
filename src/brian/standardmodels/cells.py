"""
Standard cells for the brian module


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from pyNN.standardmodels import cells, build_translations, ModelNotAvailable
from pyNN import errors
#import brian_no_units_no_warnings
from brian.library.synapses import *
import brian
from pyNN.brian.simulator import SimpleCustomRefractoriness, AdaptiveReset
from brian import mV, ms, nF, nA, uS, second, Hz, amp
import numpy

class IF_curr_alpha(cells.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
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
    
    @property
    def threshold(self):
        return self.parameters['v_thresh'] * mV
        
    @property
    def reset(self):
        return self.parameters['v_reset'] * mV    

class IF_curr_exp(cells.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
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
    
    @property
    def threshold(self):
        return self.parameters['v_thresh'] * mV
    
    @property
    def reset(self):
        return self.parameters['v_reset'] * mV    


class IF_cond_alpha(cells.IF_cond_alpha):
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
    
    @property
    def threshold(self):
        return self.parameters['v_thresh'] * mV

    @property
    def reset(self):
        return self.parameters['v_reset'] * mV    


class IF_cond_exp(cells.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and 
    exponentially-decaying post-synaptic conductance."""
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
    
    @property
    def threshold(self):
        return self.parameters['v_thresh'] * mV

    @property
    def reset(self):
        return self.parameters['v_reset'] * mV    


class IF_facets_hardware1(ModelNotAvailable):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    pass


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""
    translations = build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'duration'),
    )
    
    class rates(object):
        """
        Acts as a function of time for the PoissonGroup, while storing the
        parameters for later retrieval.
        """
        def __init__(self, start, duration, rate, n):
            self.start    = start * numpy.ones(n) * ms
            self.duration = duration * numpy.ones(n) * ms
            self.rate     = rate * numpy.ones(n) * Hz
        
        def __call__(self, t):
            idx = (self.start <= t) & (t <= self.start + self.duration)
            return numpy.where(idx, self.rate, 0)
    
    def __init__(self, parameters):
        cells.SpikeSourcePoisson.__init__(self, parameters)
        start    = self.parameters['start']
        duration = self.parameters['duration']
        rate     = self.parameters['rate']    
    
class SpikeSourceArray(cells.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""
    translations = build_translations(
        ('spike_times', 'spiketimes', ms),
    )

    @classmethod
    def translate(cls, parameters):
        if 'spike_times' in parameters:
            try:
                parameters['spike_times'] = numpy.array(parameters['spike_times'], float)
            except ValueError:
                raise errors.InvalidParameterValueError("spike times must be floats")
        return super(SpikeSourceArray, cls).translate(parameters)


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr, EIF_cond_exp_isfa_ista
    """
    
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
    
    @property
    def threshold(self):
        return self.parameters['v_spike'] * mV
        
    @property
    def reset(self):
        reset = AdaptiveReset(self.parameters['v_reset'] * mV, self.parameters['b'] * amp)
        return SimpleCustomRefractoriness(reset, period = self.parameters['tau_refrac'] * ms)

class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr, EIF_cond_exp_isfa_ista
    """
    
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
    
    @property
    def threshold(self):
        return self.parameters['v_spike'] * mV
        
    @property
    def reset(self):
        reset = AdaptiveReset(self.parameters['v_reset'] * mV, self.parameters['b'] * amp)
        return SimpleCustomRefractoriness(reset, period = self.parameters['tau_refrac'] * ms)
        

class HH_cond_exp(cells.HH_cond_exp):
   
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
   
   @property
   def threshold(self):
       return brian.EmpiricalThreshold(threshold=-40*mV, refractory=2*ms)

   @property
   def reset(self):
       return 0 * mV

   @property 
   def extra(self):
       return {'implicit' : True}

class SpikeSourceInhGamma(ModelNotAvailable):
    pass


class IF_cond_exp_gsfa_grr(ModelNotAvailable):
    pass
