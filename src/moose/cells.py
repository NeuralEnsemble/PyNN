from pyNN import common
class HH_cond_exp(common.HH_cond_exp):
    """Single compartment cell with an Na channel and a K channel"""
    translations = common.build_translations(
        ('gbar_Na',    'GbarNa', 1e-9),   
        ('gbar_K',     'GbarK', 1e-9),    
        ('g_leak',     'GLeak', 1e-9),    
        ('cm',         'Cm',    1e-9),  
        ('v_offset',   'Voff', 1e-3),
        ('e_rev_Na',   'ENa', 1e-3),
        ('e_rev_K',    'EK', 1e-3), 
        ('e_rev_leak', 'Vleak', 1e-3),
        ('e_rev_E',    'ESynE', 1e-3),
        ('e_rev_I',    'ESynI', 1e-3),
        ('tau_syn_E',  'tauE', 1e-3),
        ('tau_syn_I',  'tauI', 1e-3),
        ('i_offset',   'inject', 1e-9),
        ('v_init',     'initVm', 1e-3),
    )
    moose_name = "SingleCompHH"











