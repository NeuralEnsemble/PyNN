# ==============================================================================
# Standard cells for nest1
# $Id: cells.py 294 2008-04-04 12:07:56Z apdavison $
# ==============================================================================

from pyNN import common
import brian_no_units_no_warnings
from brian.library.synapses import *
import brian


class IF_curr_alpha(common.IF_curr_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current."""
    translations = common.build_translations(
        ('v_rest',     'v_rest', 0.001),
        ('v_reset',    'v_reset', 0.001),
        ('cm',         'cm'), # C is in pF, cm in nF
        ('tau_m',      'tau_m', 0.001),
        ('tau_refrac', 'tau_refrac', "max(get_time_step(), tau_refrac)", "tau_refrac"),
        ('tau_syn_E',  'tau_syn_E', 0.001),
        ('tau_syn_I',  'tau_syn_I', 0.001),
        ('v_thresh',   'v_thresh', 0.001),
        ('i_offset',   'i_offset'), # I0 is in pA, i_offset in nA
        ('v_init',     'v', 0.001),
    )
    eqs= brian.Equations('''
        dv/dt  = (ge + gi-(v-v_rest))/tau_m : volt
        dge/dt = (y-ge)/tau_syn_E           : volt 
        dy/dt = -y/tau_syn_E                : volt
        dgi/dt = (y-gi)/tau_syn_I           : volt 
        dy/dt = -y/tau_syn_I                : volt
        tau_syn_E : second
        tau_syn_I : second
        tau_m     : second
        v_rest    : volt
        '''
        )


class IF_curr_exp(common.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = common.build_translations(
        ('v_rest',     'v_rest', 0.001),
        ('v_reset',    'v_reset', 0.001),
        ('cm',         'cm'), # C is in pF, cm in nF
        ('tau_m',      'tau_m', 0.001),
        ('tau_refrac', 'tau_refrac',  "max(get_time_step(), tau_refrac)*0.001", "tau_refrac"),
        ('tau_syn_E',  'tau_syn_E', 0.001),
        ('tau_syn_I',  'tau_syn_I', 0.001),
        ('v_thresh',   'v_thresh', 0.001),
        ('i_offset',   'i_offset'), # I0 is in pA, i_offset in nA
        ('v_init',     'v', 0.001),
    )
    eqs= brian.Equations('''
        dv/dt  = (ge + gi-(v-v_rest))/tau_m : volt
        dge/dt = -ge/tau_syn_E              : volt
        dgi/dt = -gi/tau_syn_I              : volt
        tau_syn_E : second
        tau_syn_I : second
        tau_m     : second
        v_rest    : volt
        '''
        )
        
    

class IF_cond_alpha(common.ModelNotAvailable):
    pass


class IF_cond_exp(common.IF_cond_exp):
    """Leaky integrate and fire model with fixed threshold and 
    exponentially-decaying post-synaptic conductance."""
    pass
    

class IF_facets_hardware1(common.IF_facets_hardware1):
    """Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1. For further
    details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """
    pass


class SpikeSourcePoisson(common.SpikeSourcePoisson):
    """Spike source, generating spikes according to a Poisson process."""
    translations = common.build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'duration'),
    )
    
class SpikeSourceArray(common.SpikeSourceArray):
    """Spike source generating spikes at the times given in the spike_times array."""
    pass

    
class EIF_cond_alpha_isfa_ista(common.ModelNotAvailable):
    pass

class HH_cond_exp(common.ModelNotAvailable):
    pass

class SpikeSourceInhGamma(common.ModelNotAvailable):
    pass

class IF_cond_exp_gsfa_grr(common.ModelNotAvailable):
    pass