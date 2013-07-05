"""
Standard cells for pcsim

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import common, errors
from pyNN.standardmodels import cells, build_translations, ModelNotAvailable
import pypcsim
import numpy
import logging

logger = logging.getLogger("PyNN")

class IF_curr_alpha(cells.IF_curr_alpha):
    
    __doc__ = cells.IF_curr_alpha.__doc__   
    
    translations = build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
    )
    pcsim_name = "LIFCurrAlphaNeuron"    
    simObjFactory = None
    setterMethods = {}
        
    def __init__(self, parameters):
        cells.IF_curr_alpha.__init__(self, parameters)              
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = pypcsim.LIFCurrAlphaNeuron(**self.parameters)


class IF_curr_exp(cells.IF_curr_exp):
    
    __doc__ = cells.IF_curr_exp.__doc__    
    
    translations = build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3), 
    )
    pcsim_name = "LIFCurrExpNeuron"    
    simObjFactory = None
    setterMethods = {}
    
    def __init__(self, parameters):
        cells.IF_curr_exp.__init__(self, parameters)                
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = pypcsim.LIFCurrExpNeuron(**self.parameters)


class IF_cond_alpha(cells.IF_cond_alpha):
    
    __doc__ = cells.IF_cond_alpha.__doc__    
    
    translations = build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('e_rev_E',    'ErevExc',   1e-3),
        ('e_rev_I',    'ErevInh',   1e-3),
    )
    pcsim_name = "LIFCondAlphaNeuron"    
    simObjFactory = None
    setterMethods = {}
    recordable = ['spikes', 'v']
        
    def __init__(self, parameters):
        cells.IF_cond_alpha.__init__(self, parameters)
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = pypcsim.LIFCondAlphaNeuron(**self.parameters)


class IF_cond_exp(cells.IF_cond_exp):
    
    __doc__ = cells.IF_cond_exp.__doc__    
    
    translations = build_translations(
        ('tau_m',      'taum',      1e-3),
        ('cm',         'Cm',        1e-9), 
        ('v_rest',     'Vresting',  1e-3), 
        ('v_thresh',   'Vthresh',   1e-3), 
        ('v_reset',    'Vreset',    1e-3), 
        ('tau_refrac', 'Trefract',  1e-3), 
        ('i_offset',   'Iinject',   1e-9),         
        ('tau_syn_E',  'TauSynExc', 1e-3),
        ('tau_syn_I',  'TauSynInh', 1e-3),
        ('e_rev_E',    'ErevExc',   1e-3),
        ('e_rev_I',    'ErevInh',   1e-3),
    )
    pcsim_name = "LIFCondExpNeuron"    
    simObjFactory = None
    setterMethods = {}
    recordable = ['spikes', 'v']
        
    def __init__(self, parameters):
        cells.IF_cond_exp.__init__(self, parameters)
        self.parameters['Inoise'] = 0.0
        self.simObjFactory = pypcsim.LIFCondExpNeuron(**self.parameters)


""" Implemented not tested """
class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    
    __doc__ = cells.SpikeSourcePoisson.__doc__     

    translations = build_translations(
        ('start',    'Tstart',   1e-3), 
        ('rate',     'rate'), 
        ('duration', 'duration', 1e-3)
    )
    pcsim_name = 'PoissonInputNeuron'    
    simObjFactory = None
    setterMethods = {}
   
    def __init__(self, parameters):
        cells.SpikeSourcePoisson.__init__(self, parameters)      
        self.simObjFactory = pypcsim.PoissonInputNeuron(**self.parameters)

    
def sanitize_spike_times(spike_times):
    """
    PCSIM has a bug that the SpikingInputNeuron sometimes stops emitting spikes
    I think this happens when two spikes fall in the same time step.
    This workaround removes any spikes after the first within a given time step.
    """
    time_step = common.get_time_step()
    try:
        spike_times = numpy.array(spike_times, float)
    except ValueError, e:
        raise errors.InvalidParameterValueError("Spike times must be floats. %s")
    
    bins = (spike_times/time_step).astype('int')
    mask = numpy.concatenate((numpy.array([True]), bins[1:] != bins[:-1]))
    if mask.sum() < len(bins):
        logger.warn("Spikes have been thrown away because they were too close together.")
        logger.debug(spike_times[(1-mask).astype('bool')])
    if len(spike_times) > 0:
        return spike_times[mask]
    else:
        return spike_times

class SpikeSourceArray(cells.SpikeSourceArray):

    __doc__ = cells.SpikeSourceArray.__doc__    

    translations = build_translations(
        ('spike_times', 'spikeTimes'), # 1e-3), 
    )
    pcsim_name = 'SpikingInputNeuron'
    simObjFactory = None
    setterMethods = {'spikeTimes':'setSpikes'}
    getterMethods = {'spikeTimes':'getSpikeTimes' }
    
    def __init__(self, parameters):
        cells.SpikeSourceArray.__init__(self, parameters)
        self.pcsim_object_handle = pypcsim.SpikingInputNeuron(**self.parameters)
        self.simObjFactory  = pypcsim.SpikingInputNeuron(**self.parameters)
    
    @classmethod
    def translate(cls, parameters):
        """Translate standardized model parameters to simulator-specific parameters."""
        native_parameters = super(SpikeSourceArray, cls).translate(parameters)
        native_parameters['spikeTimes'] = sanitize_spike_times(native_parameters['spikeTimes'])
        # for why we used 'super' here, see http://blogs.gnome.org/jamesh/2005/06/23/overriding-class-methods-in-python/
        # convert from ms to s - should really be done in common.py, but that doesn't handle lists, only arrays
        if isinstance(native_parameters['spikeTimes'], list):
            native_parameters['spikeTimes'] = [t*0.001 for t in native_parameters['spikeTimes']]
        elif isinstance(native_parameters['spikeTimes'], numpy.ndarray):
            native_parameters['spikeTimes'] *= 0.001
        return native_parameters
    
    @classmethod
    def reverse_translate(cls, native_parameters):
        """Translate simulator-specific model parameters to standardized parameters."""
        standard_parameters = super(SpikeSourceArray, cls).reverse_translate(native_parameters)
        if isinstance(standard_parameters['spike_times'], list):
            standard_parameters['spike_times'] = [t*1000.0 for t in standard_parameters['spike_times']]
        elif isinstance(standard_parameters['spike_times'], numpy.ndarray):
            standard_parameters['spike_times'] *= 1000.0 
        return standard_parameters

class EIF_cond_alpha_isfa_ista(ModelNotAvailable):
    pass
#class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
#    """
#    Exponential integrate and fire neuron with spike triggered and sub-threshold
#    adaptation currents (isfa, ista reps.) according to:
#    
#    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as
#    an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642
#
#    See also: IF_cond_exp_gsfa_grr
#    """
#
#    translations = build_translations(
#        ('cm'        , 'Cm',        1e-9),  # nF -> F
#        ('tau_refrac', 'Trefract',  1e-3),  # ms -> s 
#        ('v_spike'   , 'Vpeak',     1e-3),
#        ('v_reset'   , 'Vr',        1e-3),
#        ('v_rest'    , 'El',        1e-3),
#        ('tau_m'     , 'gl',        "1e-6*cm/tau_m", "Cm/gl"), # units correct?
#        ('i_offset'  , 'Iinject',   1e-9),
#        ('a'         , 'a',         1e-9),       
#        ('b'         , 'b',         1e-9),
#        ('delta_T'   , 'slope',     1e-3), 
#        ('tau_w'     , 'tau_w',     1e-3), 
#        ('v_thresh'  , 'Vt',        1e-3), 
#        ('e_rev_E'   , 'ErevExc',   1e-3),
#        ('tau_syn_E' , 'TauSynExc', 1e-3), 
#        ('e_rev_I'   , 'ErevInh',   1e-3), 
#        ('tau_syn_I' , 'TauSynInh',  1e-3),
#    )
#    pcsim_name = "aEIFCondAlphaNeuron"
#    simObjFactory = None
#    setterMethods = {}
#    
#    def __init__(self, parameters):
#        cells.EIF_cond_alpha_isfa_ista.__init__(self, parameters)              
#        self.parameters['Inoise'] = 0.0
#        limited_parameters = {} # problem that Trefract is not implemented
#        for k in ('a','b','Vt','Vr','El','gl','Cm','tau_w','slope','Vpeak',
#                  'Vinit','Inoise','Iinject', 'ErevExc', 
#                  'TauSynExc', 'ErevInh', 'TauSynInh'):
#            limited_parameters[k] = self.parameters[k]
#        self.simObjFactory = getattr(pypcsim, EIF_cond_alpha_isfa_ista.pcsim_name)(**limited_parameters)

class IF_facets_hardware1(ModelNotAvailable):
    pass

class HH_cond_exp(ModelNotAvailable):
    pass

#class HH_cond_exp(cells.HH_cond_exp):
#    """docstring needed here."""
#    
#    translations = build_translations(
#        ('gbar_Na',    'gbar_Na'),   
#        ('gbar_K',     'gbar_K'),    
#        ('g_leak',     'Rm',    '1/g_leak', '1/Rm'),    # check HHNeuronTraubMiles91.h, not sure this is right
#        ('cm',         'Cm',    1000.0),  
#        ('v_offset',   'V_T'), # PCSIM fixes V_T at -63 mV
#        ('e_rev_Na',   'E_Na'),
#        ('e_rev_K',    'E_K'), 
#        ('e_rev_leak', 'Vresting'), # check HHNeuronTraubMiles91.h, not sure this is right
#        ('e_rev_E',    'ErevExc'),
#        ('e_rev_I',    'ErevInh'),
#        ('tau_syn_E',  'TauSynExc'),
#        ('tau_syn_I',  'TauSynInh'),
#        ('i_offset',   'Iinject', 1000.0),
#    )
#    pcsim_name = "HHNeuronTraubMiles91"    
#    simObjFactory = None
#    setterMethods = {}
#        
#    def __init__(self, parameters):
#        cells.HH_cond_exp.__init__(self, parameters)
#        self.parameters['Inoise'] = 0.0
#        self.simObjFactory = pypcsim.LIFCondAlphaNeuron(**self.parameters)
        
        
        
class SpikeSourceInhGamma(ModelNotAvailable):
    pass

class IF_cond_exp_gsfa_grr(ModelNotAvailable):
    pass
