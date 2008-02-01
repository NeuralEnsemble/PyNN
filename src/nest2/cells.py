# ==============================================================================
# Standard cells for nest2
# $Id$
# ==============================================================================

from pyNN import common
 
class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""

    translations = common.build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref',  "max(dt, tau_refrac)", "tau_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0), # I_e is in pA, i_offset in nA
        ('v_init',     'V_m'),
    )
    nest_name = "iaf_psc_alpha"
    def __init__(self,parameters):
        common.IF_curr_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)


class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref_abs',  "max(dt, tau_refrac)", "tau_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0), # I_e is in pA, i_offset in nA
        ('v_init',     'V_m'),
    )
    nest_name = 'iaf_psc_exp'
    def __init__(self,parameters):
        common.IF_curr_exp.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)


class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""

    translations = common.build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref',      "max(dt, tau_refrac)", "t_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('v_init',     'V_m'),
    )
    nest_name = "iaf_cond_alpha"
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)
        

class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref',      "max(dt, tau_refrac)", "t_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('v_init',     'V_m'),
    )
    nest_name = "iaf_cond_exp"
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)
        

class IF_cond_exp_sfa_rr(common.IF_cond_exp_sfa_rr):
    """Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based spike-frequency adaptation,
    and a conductance-based relative refractory mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.

    Depreciated: Use the equivalent type 'IF_cond_exp_gsfa_grr' instead.

    """
    translations = common.build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref',      "max(dt, tau_refrac)", "t_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('v_init',     'V_m'),
        ('tau_sfa',    'tau_sfa'),
        ('e_rev_sfa',  'E_sfa'),
        ('q_sfa',      'q_sfa'),
        ('tau_rr',     'tau_rr'),
        ('e_rev_rr',   'E_rr'),
        ('q_rr',       'q_rr')
    )
    nest_name = "iaf_cond_exp_sfa_rr"
    def __init__(self,parameters):
        common.IF_cond_exp_sfa_rr.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)


class IF_cond_exp_gsfa_grr(common.IF_cond_exp_gsfa_grr):
    """Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based spike-frequency adaptation,
    and a conductance-based relative refractory mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.

    NOTE: This is a renaming if the now depreciated 'IF_cond_exp_sfa_rr'.

    See also: EIF_cond_alpha_isfa_ista

    """
    translations = common.build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref',      "max(dt, tau_refrac)", "t_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('v_init',     'V_m'),
        ('tau_sfa',    'tau_sfa'),
        ('e_rev_sfa',  'E_sfa'),
        ('q_sfa',      'q_sfa'),
        ('tau_rr',     'tau_rr'),
        ('e_rev_rr',   'E_rr'),
        ('q_rr',       'q_rr')
    )
    nest_name = "iaf_cond_exp_sfa_rr"
    def __init__(self,parameters):
        common.IF_cond_exp_gsfa_grr.__init__(self,parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)



class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed 
    threshold as it is resembled by the FACETS Hardware Stage 1. For further 
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    # in 'iaf_cond_exp_sfa_rr', the dimension of C_m is pF, 
    # while in the pyNN context, cm is given in nF
    translations = {
        'v_reset'   : ('V_reset',        "parameters['v_reset']"),
        'v_rest'    : ('E_L',            "parameters['v_rest']"),
        'v_thresh'  : ('V_th',           "parameters['v_thresh']"),
        'e_rev_E'   : ('E_ex',           "parameters['e_rev_E']"),
        'e_rev_I'   : ('E_in',           "parameters['e_rev_I']"),
        'cm'        : ('C_m',            "parameters['cm']*1000.0"), 
        'tau_refrac': ('t_ref',          "max(dt,parameters['tau_refrac'])"),
        'tau_syn_E' : ('tau_syn_ex',     "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_syn_in',     "parameters['tau_syn_I']"),                              
        'g_leak'    : ('g_L',            "parameters['g_leak']")    
    }
    nest_name = "iaf_cond_exp_sfa_rr"

    def __init__(self, parameters):
        common.IF_facets_hardware1.__init__(self,parameters)
        self.parameters = self.translate(self.parameters)
        self.parameters['q_rr']     = 0.0
        self.parameters['q_sfa']    = 0.0
        

class HH_cond_exp(common.HH_cond_exp):
    """docstring needed here."""
    
    translations = {
        'gbar_Na'   : ('g_Na',   "parameters['gbar_Na']"),   
        'gbar_K'    : ('g_K',    "parameters['gbar_K']"),    
        'g_leak'    : ('g_L',    "parameters['g_leak']"),    
        'cm'        : ('C_m',    "parameters['cm']*1000.0"),  
        'v_offset'  : ('U_tr',   "parameters['v_offset']"),
        'e_rev_Na'  : ('E_Na',   "parameters['e_rev_Na']"),
        'e_rev_K'   : ('E_K',    "parameters['e_rev_K']"), 
        'e_rev_leak': ('E_L',    "parameters['e_rev_leak']"),
        'e_rev_E'   : ('E_ex',   "parameters['e_rev_E']"),
        'e_rev_I'   : ('E_in',   "parameters['e_rev_I']"),
        'tau_syn_E' : ('tau_ex', "parameters['tau_syn_E']"),
        'tau_syn_I' : ('tau_in', "parameters['tau_syn_I']"),
        'i_offset'  : ('I_stim', "parameters['i_offset']*1000.0"),
        'v_init'    : ('V_m',    "parameters['v_init']"),
    }
    nest_name = "hh_cond_exp_traub"
    
    def __init__(self,parameters):
        common.HH_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                     # values for not-specified parameters.
        self.parameters = self.translate(self.parameters)
        
        
class AdaptiveExponentialIF_alpha(common.AdaptiveExponentialIF_alpha):
    """adaptive exponential integrate and fire neuron according to 
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    Depreciated: Use the equivalent type 'EIF_cond_alpha_isfa_ista' instead.

    """

    translations = common.build_translations(
        ('v_init'    , 'V_m'),
        ('w_init'    , 'w',         1000.0),  # nA -> pA
        ('cm'        , 'C_m',       1000.0),  # nF -> pF
        ('tau_refrac', 't_ref'), 
        ('v_spike'   , 'V_peak'),
        ('v_reset'   , 'V_reset'),
        ('v_rest'    , 'E_L'),
        ('tau_m'     , 'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('i_offset'  , 'I_e',       1000.0),  # nA -> pA
        ('a'         , 'a'),       
        ('b'         , 'b',         1000.0),  # nA -> pA.
        ('delta_T'   , 'Delta_T'), 
        ('tau_w'     , 'tau_w'), 
        ('v_thresh'  , 'V_th'), 
        ('e_rev_E'   , 'E_ex'),
        ('tau_syn_E' , 'tau_syn_ex'), 
        ('e_rev_I'   , 'E_in'), 
        ('tau_syn_I' , 'tau_syn_in'),
    )
    nest_name = "aeif_cond_alpha"
    
    def __init__(self,parameters):
        common.AdaptiveExponentialIF_alpha.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)
        


class EIF_cond_alpha_isfa_ista(common.EIF_cond_alpha_isfa_ista):
    """exponential integrate and fire neuron with spike triggered and sub-threshold
    adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    NOTE: This is a renaming if the now depreciated 'AdaptiveExponentialIF_alpha'.

    See also: IF_cond_exp_gsfa_grr

    """

    translations = common.build_translations(
        ('v_init'    , 'V_m'),
        ('w_init'    , 'w',         1000.0),  # nA -> pA
        ('cm'        , 'C_m',       1000.0),  # nF -> pF
        ('tau_refrac', 't_ref'), 
        ('v_spike'   , 'V_peak'),
        ('v_reset'   , 'V_reset'),
        ('v_rest'    , 'E_L'),
        ('tau_m'     , 'g_L',       "cm/tau_m*1000.0", "C_m/g_L"),
        ('i_offset'  , 'I_e',       1000.0),  # nA -> pA
        ('a'         , 'a'),       
        ('b'         , 'b',         1000.0),  # nA -> pA.
        ('delta_T'   , 'Delta_T'), 
        ('tau_w'     , 'tau_w'), 
        ('v_thresh'  , 'V_th'), 
        ('e_rev_E'   , 'E_ex'),
        ('tau_syn_E' , 'tau_syn_ex'), 
        ('e_rev_I'   , 'E_in'), 
        ('tau_syn_I' , 'tau_syn_in'),
    )
    nest_name = "aeif_cond_alpha"
    
    def __init__(self,parameters):
        common.EIF_cond_alpha_isfa_ista.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)
        
        
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


class SpikeSourceInhGamma(common.SpikeSourceInhGamma):
    """Spike source, generating realizations of an inhomogeneous gamma process, employing
    the thinning method.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.
    """

    translations = {
        'a'     : ('a'   , "parameters['a']"),
        'b'     : ('b'   , "parameters['b']"),
        'tbins'     : ('tbins'   , "parameters['tbins']"),
        'rmax'     : ('rmax'   , "parameters['rmax']"),
        'start'    : ('start'  , "parameters['start']"),
        'duration' : ('stop'   , "parameters['duration']+parameters['start']")
    }
    nest_name = 'inh_gamma_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceInhGamma.__init__(self,parameters)
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
    
