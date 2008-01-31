# ==============================================================================
# Standard cells for neuron
# $Id: cells.py 191 2008-01-29 10:36:00Z apdavison $
# ==============================================================================

from pyNN import common
from pyNN.neuron2 import neuron
from math import pi

class StandardIF(neuron.nrn.Section):
    """docstring"""
    
    synapse_models = {
        'current':      { 'exp': neuron.ExpISyn, 'alpha': neuron.AlphaISyn },
        'conductance' : {'exp': neuron.ExpSyn, 'alpha': neuron.AlphaSyn },
    }
    
    def __init__(self, syn_type, syn_shape, tau_m=20, cm=1.0, v_rest=-65,
                 v_thresh=-55, t_refrac=2, i_offset=0, v_reset=None,
                 v_init=None, tau_e=5, tau_i=5, e_e=0, e_i=-70):

        # initialise Section object with 'pas' mechanism
        neuron.nrn.Section.__init__(self)
        self.seg = self(0.5)
        self.L = 100
        self.seg.diam = 1000/math.pi # gives area = 1e-3 cm2
        self.insert('pas')
        
        # insert synapses
        assert syn_type in ('current', 'conductance'), "syn_type must be either 'current' or 'conductance'"
        assert syn_shape in ('alpha', 'exp'), "syn_type must be either 'alpha' or 'exp'"
        synapse_model = synapse_models[syn_type][syn_shape]
        if syn_type == 'current':
            esyn = synapse_model(self, 0.5, tau=tau_e)
            isyn = synapse_model(self, 0.5, tau=tau_i)
        elif syn_type == 'conductance':
            esyn = synapse_model(self, 0.5, tau=tau_e, e=e_e)
            isyn = synapse_model(self, 0.5, tau=tau_i, e=e_i)    

        # insert current source
        stim = neuron.IClamp(self, 0.5, delay=0, dur=1e12, amp=i_offset)
        
        # process arguments
        for name in ('tau_m', 'cm', 'v_rest', 'v_thresh', 't_refrac',
                     'i_offset', 'v_reset', 'v_init', 'tau_e', 'tau_i'):
            setattr(self, name, locals()[name])
        if self.v_reset is None:
            self.v_reset = self.v_rest
        if self.v_init is None:
            self.v_init = self.v_rest
        if syn_type == 'conductance':
            self.e_e = e_e
            self.e_i = e_i
            
        # need to deal with FinitializeHandlers ??
        #fih = new FInitializeHandler("memb_init()",this)
        #fih2 = new FInitializeHandler("param_update()", this)
        
        

    def __property_factory(self, name, mechanism=None):
        def set(self, value):
            if mechanism:
                setattr(getattr(self.seg, mechanism), name, value)
            else:
                setattr(self.seg, name, value)
        def get(self):
            return getattr(self.seg.pas, name)
        return property(set, get)
    
    def __set_tau_m(self, value):
        self.seg.pas.g = 1e-3*self.seg.cm/value # cm(nF)/tau_m(ms) = G(uS) = 1e-6G(S). Divide by area (1e-3) to get factor of 1e-3
        
    def __get_tau_m(self):
        return 1e-3*self.seg.cm/self.seg.pas.g
    
    tau_m = property(__set_tau_m, __get_tau_m)
    v_rest = self.__property_factory('e', 'pas')
    cm = self.__property_factory('cm')
    # need properties for tau_e, e_e, i_stim, etc

class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'alpha'

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'current'
        self.parameters['syn_shape'] = 'exp'


class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
        'e_rev_E'   : ('e_e'      , "parameters['e_rev_E']"),
        'e_rev_I'   : ('e_i'      , "parameters['e_rev_I']")
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'alpha'


class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and 
    decaying-exponential post-synaptic conductance."""
    
    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'i_offset'  : ('i_offset' , "parameters['i_offset']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
        'e_rev_E'   : ('e_e'      , "parameters['e_rev_E']"),
        'e_rev_I'   : ('e_i'      , "parameters['e_rev_I']")
    }
    hoc_name = "StandardIF"
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'exp'

class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """

    translations = {
        'tau_m'     : ('tau_m'    , "parameters['tau_m']"),
        'cm'        : ('CM'       , "parameters['cm']"),
        'v_rest'    : ('v_rest'   , "parameters['v_rest']"),
        'v_thresh'  : ('v_thresh' , "parameters['v_thresh']"),
        'v_reset'   : ('v_reset'  , "parameters['v_reset']"),
        'tau_refrac': ('t_refrac' , "parameters['tau_refrac']"),
        'g_leak'    : ('tau_m'    , "parameters['cm']*1000./parameters['g_leak']"),
        'tau_syn_E' : ('tau_e'    , "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_i'    , "parameters['tau_syn_I']"),
        'v_init'    : ('v_init'   , "parameters['v_init']"),
        'e_rev_E'   : ('e_e'      , "parameters['e_rev_E']"),
        'e_rev_I'   : ('e_i'      , "parameters['e_rev_I']")
    }
    hoc_name = "StandardIF"

    def __init__(self,parameters):
        common.IF_facets_hardware1.__init__(self,parameters)

        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'exp'
        self.parameters['i_offset']  = 0.0


class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'start'    : ('start'  , "parameters['start']"),
        'rate'     : ('number' , "int((parameters['rate']/1000.0)*parameters['duration'])"),
        'duration' : ('number' , "int((parameters['rate']/1000.0)*parameters['duration'])")
    }
    hoc_name = 'SpikeSource'
   
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['source_type'] = 'NetStim'    
        self.parameters['noise'] = 1

    def translate(self,parameters):
        translated_parameters = common.SpikeSourcePoisson.translate(self,parameters)
        if parameters.has_key('rate') and parameters['rate'] != 0:
            translated_parameters['interval'] = 1000.0/parameters['rate']
        return translated_parameters

class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = {
        'spike_times' : ('spiketimes' , "parameters['spike_times']"),
    }
    hoc_name = 'SpikeSource'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)  
        self.parameters['source_type'] = 'VecStim'
        
class AdaptiveExponentialIF_alpha(common.AdaptiveExponentialIF_alpha):
    """Adaptive exponential integrate and fire neuron according to Brette and Gerstner (2005)"""
    
    translations = {
        'v_init'    : ('v_init',   "parameters['v_init']"),
        'w_init'    : ('w_init',   "parameters['w_init']"),
        'cm'        : ('CM',       "parameters['cm']"),
        'tau_refrac': ('Ref',      "parameters['tau_refrac']"), 
        'v_spike'   : ('Vspike',     "parameters['v_spike']"),
        'v_reset'   : ('Vbot',     "parameters['v_reset']"),
        'v_rest'    : ('EL',       "parameters['v_rest']"),
        'tau_m'     : ('GL',       "parameters['cm']/parameters['tau_m']"), # uS
        'i_offset'  : ('i_offset', "parameters['i_offset']"), 
        'a'         : ('a',        "parameters['a']*0.001"), # nS --> uS
        'b'         : ('b',        "parameters['b']"),
        'delta_T'   : ('delta',    "parameters['delta_T']"), 
        'tau_w'     : ('tau_w',    "parameters['tau_w']"), 
        'v_thresh'  : ('Vtr',      "parameters['v_thresh']"), 
        'e_rev_E'   : ('e_e',      "parameters['e_rev_E']"),
        'tau_syn_E' : ('tau_e',    "parameters['tau_syn_E']"), 
        'e_rev_I'   : ('e_i',      "parameters['e_rev_I']"), 
        'tau_syn_I' : ('tau_i',    "parameters['tau_syn_I']"),
    }
    hoc_name = "IF_BG_alpha"
    
    def __init__(self,parameters):
        common.AdaptiveExponentialIF_alpha.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['syn_type']  = 'conductance'
        self.parameters['syn_shape'] = 'alpha'