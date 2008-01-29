# ==============================================================================
#   Standard cells for nest1
# ==============================================================================

from pyNN import common

# ==============================================================================
#   Standard cells
# ==============================================================================
 
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    
    translations = {
            'v_rest'    : ('U0'    ,  "parameters['v_rest']"),
            'v_reset'   : ('Vreset',  "parameters['v_reset']"),
            'cm'        : ('C'     ,  "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('Tau'   ,  "parameters['tau_m']"),
            'tau_refrac': ('TauR'  ,  "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('TauSynE', "parameters['tau_syn_E']"),
            'tau_syn_I' : ('TauSynI', "parameters['tau_syn_I']"),
            'v_thresh'  : ('Theta' ,  "parameters['v_thresh']"),
            'i_offset'  : ('I0'    ,  "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'v_init'    : ('u'     ,  "parameters['v_init']"),
    }
    nest_name = "iaf_neuron2"
    
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)

class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = {
        'v_rest'    : ('U0'     , "parameters['v_rest']"),
        'v_reset'   : ('Vreset' , "parameters['v_reset']"),
        'cm'        : ('C'      , "parameters['cm']*1000.0"), # C is in pF, cm in nF
        'tau_m'     : ('Tau'    , "parameters['tau_m']"),
        'tau_refrac': ('TauR'   , "max(dt,parameters['tau_refrac'])"),
        'tau_syn_E' : ('TauSynE', "parameters['tau_syn_E']"),
        'tau_syn_I' : ('TauSynI', "parameters['tau_syn_I']"),
        'v_thresh'  : ('Theta'  , "parameters['v_thresh']"),
        'i_offset'  : ('I0'     , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
        'v_init'    : ('u'      , "parameters['v_init']"),
    }
    nest_name = 'iaf_exp_neuron2'
    
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)

class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = {
            'v_rest'    : ('U0'          , "parameters['v_rest']"),
            'v_reset'   : ('Vreset'      , "parameters['v_reset']"),
            'cm'        : ('C'           , "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('Tau'         , "parameters['tau_m']"),
            'tau_refrac': ('TauR'        , "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('TauSyn_E'    , "parameters['tau_syn_E']"),
            'tau_syn_I' : ('TauSyn_I'    , "parameters['tau_syn_I']"),
            'v_thresh'  : ('Theta'       , "parameters['v_thresh']"),
            'i_offset'  : ('Istim'       , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'e_rev_E'   : ('V_reversal_E', "parameters['e_rev_E']"),
            'e_rev_I'   : ('V_reversal_I', "parameters['e_rev_I']"),
            'v_init'    : ('u'           , "parameters['v_init']"),
    }
    nest_name = "iaf_cond_neuron"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['gL'] = self.parameters['C']/self.parameters['Tau'] # Trick to fix the leak conductance


class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = {
            'v_rest'    : ('U0'          , "parameters['v_rest']"),
            'v_reset'   : ('Vreset'      , "parameters['v_reset']"),
            'cm'        : ('C'           , "parameters['cm']*1000.0"), # C is in pF, cm in nF
            'tau_m'     : ('Tau'         , "parameters['tau_m']"),
            'tau_refrac': ('TauR'        , "max(dt,parameters['tau_refrac'])"),
            'tau_syn_E' : ('TauSyn_E'    , "parameters['tau_syn_E']"),
            'tau_syn_I' : ('TauSyn_I'    , "parameters['tau_syn_I']"),
            'v_thresh'  : ('Theta'       , "parameters['v_thresh']"),
            'i_offset'  : ('Istim'       , "parameters['i_offset']*1000.0"), # I0 is in pA, i_offset in nA
            'e_rev_E'   : ('V_reversal_E', "parameters['e_rev_E']"),
            'e_rev_I'   : ('V_reversal_I', "parameters['e_rev_I']"),
            'v_init'    : ('u'           , "parameters['v_init']"),
    }
    nest_name = "iaf_cond_exp"
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        self.parameters['gL'] = self.parameters['C']/self.parameters['Tau'] # Trick to fix the leak conductance


class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    # in 'iaf_sfa_neuron', the dimension of C is pF,
    # while in the pyNN context, cm is given in nF
    translations = {
        'v_reset'   : ('Vreset',        "parameters['v_reset']"),
        'v_rest'    : ('U0',            "parameters['v_rest']"),
        'v_thresh'  : ('Theta',         "parameters['v_thresh']"),
        'e_rev_E'   : ('V_reversal_E',  "parameters['e_rev_E']"),
        'e_rev_I'   : ('V_reversal_I',  "parameters['e_rev_I']"),
        'cm'        : ('C',             "parameters['cm']*1000.0"),
        'tau_refrac': ('TauR',          "max(dt,parameters['tau_refrac'])"),
        'tau_syn_E' : ('TauSyn_E',      "parameters['tau_syn_E']"),
        'tau_syn_I' : ('TauSyn_I',      "parameters['tau_syn_I']"),
        'g_leak'    : ('gL',            "parameters['g_leak']")
    }
    nest_name = "iaf_sfa_neuron"

    def __init__(self, parameters):
        common.IF_facets_hardware1.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['q_relref'] = 0.0
        self.parameters['q_sfa']    = 0.0
        self.parameters['python']   = True


class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = {
        'rate'     : ('rate'   , "parameters['rate']"),
        'start'    : ('start'  , "parameters['start']"),
        'duration' : ('stop'   , "parameters['duration']+parameters['start']")
    }
    nest_name = 'poisson_generator'
    
    
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['origin'] = 1.0
    
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = {
        'spike_times' : ('spike_times' , "parameters['spike_times']"),
    }
    nest_name = 'spike_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)  
    