# ==============================================================================
# Synapse Dynamics classes for nest2
# $Id$
# ==============================================================================

from pyNN import common


class SynapseDynamics(common.SynapseDynamics):
    """
    For specifying synapse short-term (faciliation,depression) and long-term
    (STDP) plasticity. To be passed as the `synapse_dynamics` argument to
    `Projection.__init__()` or `connect()`.
    """
    
    def __init__(self, fast=None, slow=None):
        common.SynapseDynamics.__init__(self, fast, slow)


class STDPMechanism(common.STDPMechanism):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None):
        common.STDPMechanism.__init__(self, timing_dependence, weight_dependence, voltage_dependence)


class TsodkysMarkramMechanism(common.TsodkysMarkramMechanism):
    
    def __init__(self, U, D, F, u0, r0, f0):
        common.TsodkysMarkramMechanism.__init__(self, U, D, F, u0, r0, f0)


class AdditiveWeightDependence(common.AdditiveWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    
    translations = common.build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse',]) #'stdp_synapse_hom'
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        common.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = locals()
        parameters.pop('self')
        self.parameters = self.translate1(parameters)
        self.parameters['mu_plus'] = 0.0
        self.parameters['mu_minus'] = 0.0


class MultiplicativeWeightDependence(common.MultiplicativeWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    translations = common.build_translations(
        ('w_max',     'wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse',]) #'stdp_synapse_hom'
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        common.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = locals()
        parameters.pop('self') 
        self.parameters = self.translate1(parameters)
        self.parameters['mu_plus'] = 1.0
        self.parameters['mu_minus'] = 1.0


class SpikePairRule(common.SpikePairRule):
    
    translations = common.build_translations(
        ('tau_plus',  'tau_plus'),
        ('tau_minus', 'Tau_minus'), # defined in post-synaptic neuron??!! help!
    )
    possible_models = set(['stdp_synapse',]) #'stdp_synapse_hom'
    
    def __init__(self, tau_plus, tau_minus):
        common.SpikePairRule.__init__(self, tau_plus, tau_minus)
        parameters = locals()
        parameters.pop('self')
        self.parameters = self.translate1(parameters)
        