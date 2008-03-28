# ==============================================================================
# Synapse Dynamics classes for neuron
# $Id: synapses.py 191 2008-01-29 10:36:00Z apdavison $
# ==============================================================================

from pyNN import common

class SynapseDynamics(common.SynapseDynamics):
    """
    For specifying synapse short-term (faciliation, depression) and long-term
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
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        common.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)

class MultiplicativeWeightDependence(common.MultiplicativeWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        pass

class SpikePairRule(common.SpikePairRule):
    
    def __init__(self, tau_plus, tau_minus):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus