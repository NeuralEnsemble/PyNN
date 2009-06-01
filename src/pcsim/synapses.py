"""
Synapse Dynamics classes for pcsim

$Id$
"""

from pyNN import common, synapses
import pypcsim

synapse_models = [s for s in dir(pypcsim) if 'Synapse' in s]

def get_synapse_models(criterion):
    return set([s for s in synapse_models if criterion in s])

conductance_based_synapse_models = get_synapse_models("Cond")
current_based_synapse_models = get_synapse_models("Curr")
alpha_function_synapse_models = get_synapse_models("Alpha")
double_exponential_synapse_models = get_synapse_models("DoubleExp")
single_exponential_synapse_models = get_synapse_models("Exp").difference(double_exponential_synapse_models)
#stdp_synapse_models = get_synapse_models("Stdp")
stdp_synapse_models = set(["StaticStdpSynapse",  # CurrExp
                           "StaticStdpCondExpSynapse",
                           "DynamicStdpSynapse", # CurrExp
                           "DynamicStdpCondExpSynapse"])
#dynamic_synapse_models = get_synapse_models("Dynamic")
dynamic_synapse_models = set(["DynamicSpikingSynapse", # DynamicCurrExpSynapse
                              "DynamicCondExpSynapse",
                              "DynamicCurrAlphaSynapse",
                              "DynamicCondAlphaSynapse",
                              "DynamicStdpSynapse", # CurrExp
                              "DynamicStdpCondExpSynapse",
                              # there don't seem to be any alpha-function STDP synapse models
                             ])

SynapseDynamics = common.SynapseDynamics
        
class STDPMechanism(common.STDPMechanism):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0, model=None):
        # not sure what the situation is with dendritic_delay_fraction in PCSIM
        common.STDPMechanism.__init__(self, timing_dependence, weight_dependence,
                                      voltage_dependence, dendritic_delay_fraction,model)


class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    
    translations = common.build_translations(
        ('U', 'U'),
        ('tau_rec', 'D', 1e-3),
        ('tau_facil', 'F', 1e-3),
        ('u0', 'u0'),  
        ('x0', 'r0' ), # I'm not at all sure this 
        ('y0', 'f0')   # translation is correct
                       # need to look at the source code
    )
    #possible_models = get_synapse_models("Dynamic")
    possible_models = dynamic_synapse_models
    
    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        #synapses.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)
        parameters = dict(locals()) # need the dict to get a copy of locals. When running
        parameters.pop('self')      # through coverage.py, for some reason, the pop() doesn't have any effect
        self.parameters = self.translate(parameters)
        

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    
    translations = common.build_translations(
        ('w_max',     'Wex',  1e-9), # unit conversion. This exposes a limitation of the current
                                     # translation machinery, because this value depends on the
                                     # type of the post-synaptic cell. We currently work around
                                     # this using the "scales_with_weight" attribute, although
                                     # this breaks reverse translation.
        ('w_min',     'w_min_always_zero_in_PCSIM'),
        ('A_plus',    'Apos', '1e-9*A_plus*w_max', '1e9*Apos/w_max'),  # note that here Apos and Aneg
        ('A_minus',   'Aneg', '-1e-9*A_minus*w_max', '-1e9*Aneg/w_max'), # have the same units as the weight
    )
    possible_models = stdp_synapse_models
    scales_with_weight = ['Wex', 'Apos', 'Aneg']
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by PCSIM.")
        #synapses.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['useFroemkeDanSTDP'] = False
        self.parameters['mupos'] = 0.0
        self.parameters['muneg'] = 0.0
        self.parameters.pop('w_min_always_zero_in_PCSIM')
    
    
class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    translations = common.build_translations(
        ('w_max',     'Wex',  1e-9), # unit conversion
        ('w_min',     'w_min_always_zero_in_PCSIM'),
        ('A_plus',    'Apos'),     # here Apos and Aneg
        ('A_minus',   'Aneg', -1), # are dimensionless
    )
    possible_models = stdp_synapse_models
    scales_with_weight = ['Wex']
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by PCSIM.")
        #synapses.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['useFroemkeDanSTDP'] = False
        self.parameters['mupos'] = 1.0
        self.parameters['muneg'] = 1.0
        self.parameters.pop('w_min_always_zero_in_PCSIM')


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    """
    The amplitude of the weight change depends on the current weight for
    depression (Dw propto w-w_min) and is fixed for potentiation.
    """
    translations = common.build_translations(
        ('w_max',     'Wex',  1e-9), # unit conversion
        ('w_min',     'w_min_always_zero_in_PCSIM'),
        ('A_plus',    'Apos', 1e-9), # Apos has the same units as the weight
        ('A_minus',   'Aneg', -1),   # Aneg is dimensionless
    )
    possible_models = stdp_synapse_models
    scales_with_weight = ['Wex', 'Apos']
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by PCSIM.")
        #synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['useFroemkeDanSTDP'] = False
        self.parameters['mupos'] = 0.0
        self.parameters['muneg'] = 1.0
        self.parameters.pop('w_min_always_zero_in_PCSIM')

class GutigWeightDependence(synapses.GutigWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    translations = common.build_translations(
        ('w_max',     'Wex',  1e-9), # unit conversion
        ('w_min',     'w_min_always_zero_in_PCSIM'),
        ('A_plus',    'Apos'),
        ('A_minus',   'Aneg', -1),
        ('mu_plus',   'mupos'),
        ('mu_minus',  'muneg')
    )
    possible_models = stdp_synapse_models
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01, mu_plus=0.5, mu_minus=0.5): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by PCSIM.")
        #synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['useFroemkeDanSTDP'] = False
        self.parameters.pop('w_min_always_zero_in_PCSIM')
        

class SpikePairRule(synapses.SpikePairRule):
    
    translations = common.build_translations(
        ('tau_plus',  'taupos', 1e-3),
        ('tau_minus', 'tauneg', 1e-3), 
    )
    possible_models = stdp_synapse_models
    
    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        #synapses.SpikePairRule.__init__(self, tau_plus, tau_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['STDPgap'] = 0.0
        
