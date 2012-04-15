"""
Standard cells for the nemo module


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: cells.py 897 2011-01-13 12:47:23Z pierre $
"""

from pyNN.standardmodels import cells, build_translations, ModelNotAvailable, StandardCellType
from pyNN import errors
import numpy


class Izikevich(cells.Izikevich):
    
    __doc__ = cells.Izikevich.__doc__ 

    translations = build_translations(
        ('a',    'a'),
        ('b',    'b'),
        ('v_reset', 'c'),
        ('d',    'd'),
        ('tau_refrac', 'tau_refrac')
    )

    nemo_name = "Izhikevich"

    indices = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3}
    initial_indices = {'u' : 0, 'v' : 1}



class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    
    __doc__ = cells.SpikeSourcePoisson.__doc__ 

    translations = build_translations(
        ('rate', 'rate', 0.001),
        ('start', 'start'),
        ('duration', 'duration')
    )

    nemo_name = "PoissonSource"

    indices = {'rate' : 0}



class SpikeSourceArray(cells.SpikeSourceArray):

    __doc__ = cells.SpikeSourceArray.__doc__    

    translations = build_translations(
        ('spike_times', 'spike_times'),
    )
    nemo_name = "Input"


    class spike_player(object):
        
        def __init__(self, spike_times=[], precision=1):
            self.spike_times = precision * numpy.round(spike_times/precision)        
            self.spike_times = numpy.unique(numpy.sort(self.spike_times))
            self.cursor      = 0
            self.N           = len(self.spike_times)

        @property
        def next_spike(self):
            if self.cursor < self.N:
                return self.spike_times[self.cursor]
            else:
                return numpy.inf
        
        def update(self):
            self.cursor += 1

        def reset(self, spike_times, precision):
            self.spike_times = precision * numpy.round(spike_times/precision)
            self.spike_times = numpy.unique(numpy.sort(self.spike_times))
            self.N           = len(self.spike_times)
            self.cursor      = 0

    def __init__(self, parameters):
        cells.SpikeSourceArray.__init__(self, parameters)        


class IF_cond_exp_gsfa_grr(ModelNotAvailable):
    pass

class IF_curr_alpha(cells.IF_curr_alpha):
    
    __doc__ = cells.IF_curr_alpha.__doc__    

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_reset',    'v_reset'),
        ('cm',         'cm'), 
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset'), 
    )

    indices = {
            'v_rest' : 0,
            'cm' : 2,
            'tau_m' : 3,
            't_refrac' : 4,
            'tau_syn_E' : 5,
            'tau_syn_I' : 6,
            'i_offset' : 8,
            'v_reset' : 1,
            'v_thresh' : 7
        }

    initial_indices = {'v' : 0, 'ie' : 1, 'ii' : 2}
    nemo_name = "IF_curr_alpha"


class IF_curr_exp(cells.IF_curr_exp):
    
    __doc__ = cells.IF_curr_exp.__doc__    

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_reset',    'v_reset'),
        ('cm',         'cm'), 
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset'), 
    )

    indices = {
            'v_rest' : 0,
            'cm' : 2,
            'tau_m' : 3,
            't_refrac' : 4,
            'tau_syn_E' : 5,
            'tau_syn_I' : 6,
            'i_offset' : 8,
            'v_reset' : 1,
            'v_thresh' : 7
        }

    initial_indices = {'v' : 0, 'ie' : 1, 'ii' : 2}
    nemo_name = "IF_curr_exp"


class IF_cond_alpha(cells.IF_cond_alpha):

    __doc__ = cells.IF_cond_alpha.__doc__    

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_reset',    'v_reset'),
        ('cm',         'cm'), 
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset'), 
        ('e_rev_E',    'e_rev_E'),
        ('e_rev_I',    'e_rev_I')
    )

    indices = {
            'v_rest' : 0,
            'cm' : 2,
            'tau_m' : 3,
            't_refrac' : 4,
            'tau_syn_E' : 5,
            'tau_syn_I' : 6,
            'i_offset' : 8,
            'v_reset' : 1,
            'v_thresh' : 7,
            'e_rev_E'  : 9,
            'e_rev_I'  : 10
        }

    initial_indices = {'v' : 0, 'ie' : 1, 'ii' : 2}
    nemo_name = "IF_cond_alpha"


class IF_cond_exp(cells.IF_cond_exp):
    
    __doc__ = cells.IF_cond_exp.__doc__    

    translations = build_translations(
        ('v_rest',     'v_rest'),
        ('v_reset',    'v_reset'),
        ('cm',         'cm'), 
        ('tau_m',      'tau_m'),
        ('tau_refrac', 't_refrac'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
        ('v_thresh',   'v_thresh'),
        ('i_offset',   'i_offset'), 
        ('e_rev_E',    'e_rev_E'),
        ('e_rev_I',    'e_rev_I')
    )

    indices = {
            'v_rest' : 0,
            'cm' : 2,
            'tau_m' : 3,
            't_refrac' : 4,
            'tau_syn_E' : 5,
            'tau_syn_I' : 6,
            'i_offset' : 8,
            'v_reset' : 1,
            'v_thresh' : 7,
            'e_rev_E'  : 9,
            'e_rev_I'  : 10
        }

    initial_indices = {'v' : 0, 'ie' : 1, 'ii' : 2}
    nemo_name = "IF_cond_exp"


class IF_facets_hardware1(ModelNotAvailable):
    pass

class EIF_cond_alpha_isfa_ista(ModelNotAvailable):
    pass

class EIF_cond_exp_isfa_ista(ModelNotAvailable):
    pass    

class HH_cond_exp(ModelNotAvailable):
    pass

class SpikeSourceInhGamma(ModelNotAvailable):
    pass

class IF_cond_exp_gsfa_grr(ModelNotAvailable):
    pass
