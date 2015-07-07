"""
Standard cells for nest

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import cells, build_translations
from .. import simulator


class IF_curr_alpha(cells.IF_curr_alpha):

    __doc__ = cells.IF_curr_alpha.__doc__

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
    
    __doc__ = cells.IF_curr_exp.__doc__    
    
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

    __doc__ = cells.IF_cond_alpha.__doc__    

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

    __doc__ = cells.IF_cond_exp.__doc__    
    
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

    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__    

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
    
    __doc__ = cells.IF_facets_hardware1.__doc__        

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
    extra_parameters = {
        'C_m': 200.0,
        't_ref': 1.0,
        'E_ex': 0.0
    }

class HH_cond_exp(cells.HH_cond_exp):
    
    __doc__ = cells.HH_cond_exp.__doc__    
    
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

    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__ 

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

    __doc__ = cells.SpikeSourcePoisson.__doc__ 

    translations = build_translations(
        ('rate',     'rate'),
        ('start',    'start'),
        ('duration', 'stop',    "start+duration", "stop-start"),
    )
    nest_name = {"on_grid": 'poisson_generator',
                 "off_grid": 'poisson_generator_ps'}
    always_local = True
    uses_parrot = True
    extra_parameters = {
        'origin': 1.0
    }


class SpikeSourceInhGamma(cells.SpikeSourceInhGamma):
    
    __doc__ = cells.SpikeSourceInhGamma.__doc__ 

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
    extra_parameters = {
        'origin': 1.0
    }


def adjust_spike_times_forward(spike_times):
    """
    Since this cell type requires parrot neurons, we have to adjust the
    spike times to account for the transmission delay from device to
    parrot neuron.
    """
    # todo: emit warning if any times become negative
    return spike_times - simulator.state.min_delay


def adjust_spike_times_backward(spike_times):
    """
    Since this cell type requires parrot neurons, we have to adjust the
    spike times to account for the transmission delay from device to
    parrot neuron.
    """
    return spike_times + simulator.state.min_delay


class SpikeSourceArray(cells.SpikeSourceArray):

    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'spike_times',
         adjust_spike_times_forward,
         adjust_spike_times_backward),
    )
    nest_name = {"on_grid": 'spike_generator',
                 "off_grid": 'spike_generator'}
    uses_parrot = True
    always_local = True


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

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


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__
    
    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_e', 1000.0),
    )
    nest_name = {"on_grid": "izhikevich",
                 "off_grid": "izhikevich"}
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling
