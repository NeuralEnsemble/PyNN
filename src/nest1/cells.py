# ==============================================================================
# Standard cells for nest1
# $Id$
# ==============================================================================

from pyNN import common


class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
        
    translations = common.build_translations(
        ('v_rest',     'U0'),
        ('v_reset',    'Vreset'),
        ('cm',         'C',     1000.0), # C is in pF, cm in nF
        ('tau_m',      'Tau'),
        ('tau_refrac', 'TauR',  "max(dt, tau_refrac)", "TauR"),
        ('tau_syn_E',  'TauSynE'),
        ('tau_syn_I',  'TauSynI'),
        ('v_thresh',   'Theta'),
        ('i_offset',   'I0',    1000.0), # I0 is in pA, i_offset in nA
        ('v_init',     'u'),
    )
    nest_name = "iaf_neuron2"


class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('v_rest',     'U0'),
        ('v_reset',    'Vreset'),
        ('cm',         'C',     1000.0), # C is in pF, cm in nF
        ('tau_m',      'Tau'),
        ('tau_refrac', 'TauR',  "max(dt, tau_refrac)", "TauR"),
        ('tau_syn_E',  'TauSynE'),
        ('tau_syn_I',  'TauSynI'),
        ('v_thresh',   'Theta'),
        ('i_offset',   'I0',    1000.0), # I0 is in pA, i_offset in nA
        ('v_init',     'u'),
    )
    nest_name = 'iaf_exp_neuron2'


class IF_cond_alpha(common.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('v_rest',     'U0'),
        ('v_reset',    'Vreset'),
        ('cm',         'C',     1000.0), # C is in pF, cm in nF
        ('tau_m',      'Tau'),
        ('tau_refrac', 'TauR',  "max(dt, tau_refrac)", "TauR"),
        ('tau_syn_E',  'TauSyn_E'),
        ('tau_syn_I',  'TauSyn_I'),
        ('v_thresh',   'Theta'),
        ('i_offset',   'Istim',    1000.0), # I0 is in pA, i_offset in nA
        ('v_init',     'u'),
        ('e_rev_E',    'V_reversal_E'),
        ('e_rev_I',    'V_reversal_I'),
    )
    nest_name = "iaf_cond_neuron"
    
    def __init__(self,parameters):
        common.IF_cond_alpha.__init__(self, parameters) # checks supplied parameters and adds default
                                                       # values for not-specified parameters.
        self.parameters['gL'] = self.parameters['C']/self.parameters['Tau'] # Trick to fix the leak conductance


class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""
    
    translations = common.build_translations(
        ('v_rest',     'U0'),
        ('v_reset',    'Vreset'),
        ('cm',         'C',     1000.0), # C is in pF, cm in nF
        ('tau_m',      'Tau'),
        ('tau_refrac', 'TauR',  "max(dt, tau_refrac)", "TauR"),
        ('tau_syn_E',  'TauSyn_E'),
        ('tau_syn_I',  'TauSyn_I'),
        ('v_thresh',   'Theta'),
        ('i_offset',   'Istim',    1000.0), # I0 is in pA, i_offset in nA
        ('v_init',     'u'),
        ('e_rev_E',    'V_reversal_E'),
        ('e_rev_I',    'V_reversal_I'),
    )
    nest_name = "iaf_cond_exp"
    
    def __init__(self,parameters):
        common.IF_cond_exp.__init__(self,parameters) # checks supplied parameters and adds default
                                                     # values for not-specified parameters.
        self.parameters['gL'] = self.parameters['C']/self.parameters['Tau'] # Trick to fix the leak conductance


class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    # in 'iaf_sfa_neuron', the dimension of C is pF,
    # while in the pyNN context, cm is given in nF
    translations = common.build_translations(
        ('v_reset',    'Vreset'),
        ('v_rest',     'U0'),
        ('v_thresh',   'Theta'),
        ('e_rev_E',    'V_reversal_E'),
        ('e_rev_I',    'V_reversal_I'),
        ('cm',         'C',             1000.0),
        ('tau_refrac', 'TauR',          "max(dt,tau_refrac)", "TauR"),
        ('tau_syn_E',  'TauSyn_E'),
        ('tau_syn_I',  'TauSyn_I'),
        ('g_leak',     'gL'),
    )
    nest_name = "iaf_sfa_neuron"

    def __init__(self, parameters):
        common.IF_facets_hardware1.__init__(self,parameters)
        self.parameters['q_relref'] = 0.0
        self.parameters['q_sfa']    = 0.0
        self.parameters['python']   = True


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
        self.parameters['origin'] = 1.0

    
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""

    translations = common.build_translations(
        ('spike_times', 'spike_times'),
    )
    nest_name = 'spike_generator'

    
class EIF_cond_alpha_isfa_ista(common.ModelNotAvailable):
    pass

class HH_cond_exp(common.ModelNotAvailable):
    pass

class SpikeSourceInhGamma(common.ModelNotAvailable):
    pass

class IF_cond_exp_gsfa_grr(common.ModelNotAvailable):
    pass