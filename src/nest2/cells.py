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


class IF_cond_exp_gsfa_grr(common.IF_cond_exp_gsfa_grr):
    """Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based spike-frequency adaptation,
    and a conductance-based relative refractory mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.

    NOTE: This is a renaming if the now deprecated 'IF_cond_exp_sfa_rr'.

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

class IF_cond_exp_sfa_rr(IF_cond_exp_gsfa_grr):
    """Deprecated: Use the equivalent type 'IF_cond_exp_gsfa_grr' instead."""
    pass

class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed 
    threshold as it is resembled by the FACETS Hardware Stage 1. For further 
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    # in 'iaf_cond_exp_sfa_rr', the dimension of C_m is pF, 
    # while in the pyNN context, cm is given in nF
    translations = common.build_translations(
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('v_thresh',   'V_th'),
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_refrac', 't_ref',      "max(dt, tau_refrac)", "t_ref"),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('g_leak',     'g_L'),
    )
    nest_name = "iaf_cond_exp_sfa_rr"

    def __init__(self, parameters):
        common.IF_facets_hardware1.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)
        self.parameters['q_rr']     = 0.0
        self.parameters['q_sfa']    = 0.0
        

class HH_cond_exp(common.HH_cond_exp):
    """docstring needed here."""
    
    translations = common.build_translations(
        ('gbar_Na',    'g_Na'),   
        ('gbar_K',     'g_K'),    
        ('g_leak',     'g_L'),    
        ('cm',         'C_m',    1000.0),  
        ('v_offset',   'U_tr'),
        ('e_rev_Na',   'E_Na'),
        ('e_rev_K',    'E_K'), 
        ('e_rev_leak', 'E_L'),
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_E',  'tau_ex'),
        ('tau_syn_I',  'tau_in'),
        ('i_offset',   'I_stim', 1000.0),
        ('v_init',     'V_m'),
    )
    nest_name = "hh_cond_exp_traub"
    
    def __init__(self,parameters):
        common.HH_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                     # values for not-specified parameters.
        self.parameters = self.translate1(self.parameters)
        
        
class EIF_cond_alpha_isfa_ista(common.EIF_cond_alpha_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and sub-threshold
    adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    NOTE: This is a renaming of the now deprecated 'AdaptiveExponentialIF_alpha'.

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

        
class AdaptiveExponentialIF_alpha(EIF_cond_alpha_isfa_ista):
    """Deprecated: Use the equivalent type 'EIF_cond_alpha_isfa_ista' instead."""
    pass


class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = common.build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'stop',    "start+duration", "stop-start"),
    )
    nest_name = 'poisson_generator'
    
    def __init__(self,parameters):
        common.SpikeSourcePoisson.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)
        self.parameters['origin'] = 1.0


class SpikeSourceInhGamma(common.SpikeSourceInhGamma):
    """Spike source, generating realizations of an inhomogeneous gamma process, employing
    the thinning method.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond mean-adaptation
    and renewal theories. Neural Computation 19: 2958-3010.
    """

    translations = common.build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('tbins',    'tbins'),
        ('rmax',     'rmax'),
        ('start',    'start'),
        ('duration', 'stop',   "duration+start", "stop-start"),
    )
    nest_name = 'inh_gamma_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceInhGamma.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)
        self.parameters['origin'] = 1.0


class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = common.build_translations(
        ('spike_times', 'spike_times'),
    )
    nest_name = 'spike_generator'
    
    def __init__(self,parameters):
        common.SpikeSourceArray.__init__(self,parameters)
        self.parameters = self.translate1(self.parameters)  
    
