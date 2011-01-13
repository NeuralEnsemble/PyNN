"""
Standard cells for nest

$Id$
"""

from pyNN.standardmodels import cells, build_translations
 
class IF_curr_alpha(cells.IF_curr_alpha):
    """
    Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current.
    """

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0), # I_e is in pA, i_offset in nA
    )
    nest_name = {"on_grid": "iaf_psc_alpha",
                 "off_grid": "iaf_psc_alpha"}
    standard_receptor_type = True
    

class IF_curr_exp(cells.IF_curr_exp):
    """
    Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses.
    """
    
    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',      1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',      1000.0), # I_e is in pA, i_offset in nA
    )
    nest_name = {"on_grid": 'iaf_psc_exp',
                 "off_grid": 'iaf_psc_exp_ps'}
    standard_receptor_type = True
    

class IF_cond_alpha(cells.IF_cond_alpha):
    """
    Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance.
    """

    translations = build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
    )
    nest_name = {"on_grid": "iaf_cond_alpha",
                 "off_grid": "iaf_cond_alpha"}
    standard_receptor_type = True
        

class IF_cond_exp(cells.IF_cond_exp):
    """
    Leaky integrate and fire model with fixed threshold and 
    exponentially-decaying post-synaptic conductance.
    """
    
    translations = build_translations(
        ('v_rest',     'E_L')    ,
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
    )
    nest_name = {"on_grid": "iaf_cond_exp",
                 "off_grid": "iaf_cond_exp"}
    standard_receptor_type = True


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    """
    Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based
    spike-frequency adaptation, and a conductance-based relative refractory
    mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.

    See also: EIF_cond_alpha_isfa_ista
    """
    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'V_reset'),
        ('cm',         'C_m',        1000.0), # C_m is in pF, cm in nF
        ('tau_m',      'g_L',        "cm/tau_m*1000.0", "C_m/g_L"),
        ('tau_refrac', 't_ref'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('v_thresh',   'V_th'),
        ('i_offset',   'I_e',        1000.0), # I_e is in pA, i_offset in nA
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_sfa',    'tau_sfa'),
        ('e_rev_sfa',  'E_sfa'),
        ('q_sfa',      'q_sfa'),
        ('tau_rr',     'tau_rr'),
        ('e_rev_rr',   'E_rr'),
        ('q_rr',       'q_rr')
    )
    nest_name = {"on_grid": "iaf_cond_exp_sfa_rr",
                 "off_grid": "iaf_cond_exp_sfa_rr"}
    standard_receptor_type = True


class IF_facets_hardware1(cells.IF_facets_hardware1):
    """
    Leaky integrate and fire model with conductance-based synapses and fixed 
    threshold as it is resembled by the FACETS Hardware Stage 1. For further 
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    # in 'iaf_cond_exp', the dimension of C_m is pF, 
    # while in the pyNN context, cm is given in nF
    translations = build_translations(
        ('v_reset',    'V_reset'),
        ('v_rest',     'E_L'),
        ('v_thresh',   'V_th'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('g_leak',     'g_L')
    )
    nest_name = {"on_grid": "iaf_cond_exp",
                 "off_grid": "iaf_cond_exp"}
    standard_receptor_type = True

    def __init__(self, parameters):
        cells.IF_facets_hardware1.__init__(self, parameters)
        self.parameters['C_m']   = 200.0
        self.parameters['t_ref'] =   1.0
        self.parameters['E_ex']  =   0.0


class HH_cond_exp(cells.HH_cond_exp):
    """Single-compartment Hodgkin-Huxley model."""
    
    translations = build_translations(
        ('gbar_Na',    'g_Na',  1000.0), # uS --> nS   
        ('gbar_K',     'g_K',   1000.0),
        ('g_leak',     'g_L',   1000.0),
        ('cm',         'C_m',   1000.0),  # nF --> pF
        ('v_offset',   'V_T'),
        ('e_rev_Na',   'E_Na'),
        ('e_rev_K',    'E_K'), 
        ('e_rev_leak', 'E_L'),
        ('e_rev_E',    'E_ex'),
        ('e_rev_I',    'E_in'),
        ('tau_syn_E',  'tau_syn_ex'),
        ('tau_syn_I',  'tau_syn_in'),
        ('i_offset',   'I_e',   1000.0),  # nA --> pA
    )
    nest_name = {"on_grid": "hh_cond_exp_traub",
                 "off_grid": "hh_cond_exp_traub"}
    standard_receptor_type = True
    
   
class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and sub-threshold
    adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr
    """

    translations = build_translations(
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
    nest_name = {"on_grid": "aeif_cond_alpha",
                 "off_grid": "aeif_cond_alpha"}
    standard_receptor_type = True


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""

    translations = build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'stop',    "start+duration", "stop-start"),
    )
    nest_name = {"on_grid": 'poisson_generator',
                 "off_grid": 'poisson_generator_ps'}
    always_local = True
    uses_parrot = True
    
    def __init__(self, parameters):
        cells.SpikeSourcePoisson.__init__(self, parameters)
        self.parameters['origin'] = 1.0


class SpikeSourceInhGamma(cells.SpikeSourceInhGamma):
    """
    Spike source, generating realizations of an inhomogeneous gamma process,
    employing the thinning method.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.
    """

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('tbins',    'tbins'),
        ('start',    'start'),
        ('duration', 'stop',   "duration+start", "stop-start"),
    )
    nest_name = {"on_grid": 'inh_gamma_generator',
                 "off_grid":  'inh_gamma_generator'}
    always_local = True
    
    def __init__(self, parameters):
        cells.SpikeSourceInhGamma.__init__(self, parameters)
        self.parameters['origin'] = 1.0


class SpikeSourceArray(cells.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )
    nest_name = {"on_grid": 'spike_generator',
                 "off_grid": 'spike_generator'}
    always_local = True

class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    """
    Exponential integrate and fire neuron with spike triggered and sub-threshold
    adaptation currents (isfa, ista reps.) according to:
    
    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr
    """

    translations = build_translations(
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
    nest_name = {"on_grid": "aeif_cond_exp",
                 "off_grid": "aeif_cond_exp"}
    standard_receptor_type = True
