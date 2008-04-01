# ==============================================================================
# Standard cells for pcsim
# $Id$
# ==============================================================================

from pyNN import common
from pypcsim import *
import numpy

class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = common.build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('v_init',     'Vinit',     1e-3) 
    )
    pcsim_name = "LIFCurrAlphaNeuron"    
    simObjFactory = None
    setterMethods = {}
        
    def __init__(self, parameters):
        common.IF_curr_alpha.__init__(self, parameters)              
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCurrAlphaNeuron(**self.parameters)


class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
       decaying-exponential post-synaptic current. (Separate synaptic currents for
       excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('v_init',     'Vinit',     1e-3) 
    )
    pcsim_name = "LIFCurrExpNeuron"    
    simObjFactory = None
    setterMethods = {}
    
    def __init__(self, parameters):
        common.IF_curr_exp.__init__(self, parameters)                
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCurrExpNeuron(**self.parameters)


class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('e_rev_E',    'ErevExc',   1e-3),
        ('e_rev_I',    'ErevInh',   1e-3),
        ('v_init',     'Vinit',     1e-3), 
    )
    pcsim_name = "LIFCondAlphaNeuron"    
    simObjFactory = None
    setterMethods = {}
        
    def __init__(self, parameters):
        common.IF_cond_alpha.__init__(self, parameters)
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCondAlphaNeuron(**self.parameters)


class IF_cond_exp(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and 
    exponentially-decaying post-synaptic conductance."""
    
    translations = common.build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('e_rev_E',    'ErevExc',   1e-3),
        ('e_rev_I',    'ErevInh',   1e-3),
        ('v_init',     'Vinit',     1e-3), 
    )
    pcsim_name = "LIFCondExpNeuron"    
    simObjFactory = None
    setterMethods = {}
        
    def __init__(self, parameters):
        common.IF_cond_alpha.__init__(self, parameters)
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = LIFCondExpNeuron(**self.parameters)


""" Implemented not tested """
class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = common.build_translations(
        ('start',    'Tstart',   1e-3), 
        ('rate',     'rate'), 
        ('duration', 'duration', 1e-3)
    )
    pcsim_name = 'PoissonInputNeuron'    
    simObjFactory = None
    setterMethods = {}
   
    def __init__(self, parameters):
        common.SpikeSourcePoisson.__init__(self, parameters)      
        self.simObjFactory = PoissonInputNeuron(**self.parameters)

    
""" Implemented but not tested """
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""
    translations = common.build_translations(
        ('spike_times', 'spikeTimes'), # 1e-3), 
    )
    pcsim_name = 'SpikingInputNeuron'
    simObjFactory = None
    setterMethods = {'spikeTimes':'setSpikes'}
    getterMethods = {'spikeTimes':'getSpikeTimes' }
    
    def __init__(self, parameters):
        common.SpikeSourceArray.__init__(self, parameters)
        self.pcsim_object_handle = SpikingInputNeuron(**self.parameters)
        self.simObjFactory  = SpikingInputNeuron(**self.parameters)
    
    @classmethod
    def translate(cls, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        translated_parameters = super(SpikeSourceArray, cls).translate(parameters)
        # for why we used 'super' here, see http://blogs.gnome.org/jamesh/2005/06/23/overriding-class-methods-in-python/
        # convert from ms to s - should really be done in common.py, but that doesn't handle lists, only arrays
        if isinstance(translated_parameters['spikeTimes'], list):
            translated_parameters['spikeTimes'] = [t*0.001 for t in translated_parameters['spikeTimes']]
        elif isinstance(translated_parameters['spikeTimes'], numpy.array):
            translated_parameters['spikeTimes'] *= 0.001 
        return translated_parameters
    
    @classmethod
    def reverse_translate(cls, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        standard_parameters = super(SpikeSourceArray, cls).reverse_translate(native_parameters)
        if isinstance(standard_parameters['spike_times'], list):
            standard_parameters['spike_times'] = [t*1000.0 for t in standard_parameters['spike_times']]
        elif isinstance(standard_parameters['spike_times'], numpy.ndarray):
            standard_parameters['spike_times'] *= 1000.0 
        return standard_parameters


class EIF_cond_alpha_isfa_ista(common.EIF_cond_alpha_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and sub-threshold
    adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr
    """

    translations = common.build_translations(
        ('v_init'    , 'Vinit',     1e-3),  # mV -> V
        ('w_init'    , 'w',         1e-9),  # nA -> A
        ('cm'        , 'Cm',        1e-9),  # nF -> F
        ('tau_refrac', 'Trefract',  1e-3),  # ms -> s 
        ('v_spike'   , 'Vpeak',     1e-3),
        ('v_reset'   , 'Vr',        1e-3),
        ('v_rest'    , 'El',        1e-3),
        ('tau_m'     , 'gl',        "1e-6*cm/tau_m", "Cm/gl"), # units correct?
        ('i_offset'  , 'Iinject',   1e-9),
        ('a'         , 'a',         1e-9),       
        ('b'         , 'b',         1e-9),
        ('delta_T'   , 'slope',     1e-3), 
        ('tau_w'     , 'tau_w',     1e-3), 
        ('v_thresh'  , 'Vt',        1e-3), 
        ('e_rev_E'   , 'ErevExc',   1e-3),
        ('tau_syn_E' , 'TauSynExc', 1e-3), 
        ('e_rev_I'   , 'ErevInh',   1e-3), 
        ('tau_syn_I' , 'TauSynInh',  1e-3),
    )
    pcsim_name = "CbaEIFCondAlphaNeuron"
    simObjFactory = None
    setterMethods = {}
    
    def __init__(self, parameters):
        common.EIF_cond_alpha_isfa_ista.__init__(self, parameters)              
        self.parameters['Inoise'] = 0.0
        limited_parameters = {} # problem that Trefract is not implemented
        for k in ('a','b','Vt','Vr','El','gl','Cm','tau_w','slope','Vpeak',
                  'Vinit','Inoise','Iinject', 'ErevExc', 
                  'TauSynExc', 'ErevInh', 'TauSynInh'):
            limited_parameters[k] = self.parameters[k]
        self.simObjFactory = CbaEIFCondAlphaNeuron(**limited_parameters)

class IF_facets_hardware1(common.ModelNotAvailable):
    pass

class HH_cond_exp(common.ModelNotAvailable):
    pass

class SpikeSourceInhGamma(common.ModelNotAvailable):
    pass

class IF_cond_exp_gsfa_grr(common.ModelNotAvailable):
    pass