"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""


from pyNN.standardmodels import build_translations, cells
from pyNN.moose.cells import StandardIF, SingleCompHH, RandomSpikeSource, VectorSpikeSource
from pyNN.moose.cells import mV, ms, nA, uS, nF

    


class IF_cond_exp(cells.IF_cond_exp):
    
    __doc__ = cells.IF_cond_exp.__doc__    
    
    translations = build_translations(
        ('tau_m',      'Rm',        '1e6*tau_m/cm',     '1e3*Rm*Cm'),
        ('cm',         'Cm',        nF),
        ('v_rest',     'Em',        mV),
        ('v_thresh',   'Vt',        mV),
        ('v_reset',    'Vr',        mV),
        ('tau_refrac', 'refractT',  ms),
        ('i_offset',   'inject',    nA),
        ('tau_syn_E',  'tau_e',     ms),
        ('tau_syn_I',  'tau_i',     ms),
        ('e_rev_E',    'e_e',       mV),
        ('e_rev_I',    'e_i',       mV)
    )
    model = StandardIF

    def __init__(self, parameters):
        cells.IF_cond_exp.__init__(self, parameters) # checks supplied parameters and adds default
                                                     # values for not-specified parameters.
        self.parameters['syn_shape'] = 'exp'


class IF_cond_alpha(cells.IF_cond_alpha):
    """Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance."""

    __doc__ = cells.IF_cond_alpha.__doc__        

    translations = IF_cond_exp.translations
    model = StandardIF
    
    def __init__(self, parameters):
        cells.IF_cond_alpha.__init__(self, parameters)
        self.parameters['syn_shape'] = 'alpha'


class HH_cond_exp(cells.HH_cond_exp):
    
    __doc__ = cells.HH_cond_exp.__doc__    

    translations = build_translations(
        ('gbar_Na',    'GbarNa', 1e-9),   
        ('gbar_K',     'GbarK', 1e-9),    
        ('g_leak',     'GLeak', 1e-9),    
        ('cm',         'Cm',    1e-9),  
        ('v_offset',   'Voff', 1e-3),
        ('e_rev_Na',   'ENa', 1e-3),
        ('e_rev_K',    'EK', 1e-3), 
        ('e_rev_leak', 'VLeak', 1e-3),
        ('e_rev_E',    'ESynE', 1e-3),
        ('e_rev_I',    'ESynI', 1e-3),
        ('tau_syn_E',  'tauE', 1e-3),
        ('tau_syn_I',  'tauI', 1e-3),
        ('i_offset',   'inject', 1e-9),
    )
    model = SingleCompHH



class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    
    __doc__ = cells.SpikeSourcePoisson.__doc__     

    translations = build_translations(
        ('start',    'start'),
        ('rate',     'rate'),
        ('duration', 'duration'),
    )
    model = RandomSpikeSource


class SpikeSourceArray(cells.SpikeSourceArray):
    
    __doc__ = cells.SpikeSourceArray.__doc__    

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )
    model = VectorSpikeSource







