# ==============================================================================
#   Standard cells for neuron
# ==============================================================================

from pyNN import common

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