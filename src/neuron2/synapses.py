# ==============================================================================
# Synapse Dynamics classes for neuron
# $Id: synapses.py 285 2008-04-01 15:25:22Z apdavison $
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
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        common.STDPMechanism.__init__(self, timing_dependence, weight_dependence,
                                      voltage_dependence, dendritic_delay_fraction)


class TsodyksMarkramMechanism(common.TsodyksMarkramMechanism):
    
    translations = common.build_translations(
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_facil'),
        ('u0', 'u0'),  
        ('x0', 'x' ), # } note that these two values
        ('y0', 'y')   # } are not used
    )
    native_name = 'tsodkys-markram'
    
    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        assert (x0 == 1 and y0 == 0), "It is not currently possible to set x0 and y0"
        common.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)
        parameters = locals()
        parameters.pop('self')
        self.parameters = self.translate(parameters)

class AdditiveWeightDependence(common.AdditiveWeightDependence):
    """
    The amplitude of the weight change is fixed for depression (`A_minus`)
    and for potentiation (`A_plus`).
    If the new weight would be less than `w_min` it is set to `w_min`. If it would
    be greater than `w_max` it is set to `w_max`.
    """
    
    translations = common.build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSA',])
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        common.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = locals()
        parameters.pop('self') 
        self.parameters = self.translate(parameters)


class MultiplicativeWeightDependence(common.MultiplicativeWeightDependence):
    """
    The amplitude of the weight change depends on the current weight.
    For depression, Dw propto w-w_min
    For potentiation, Dw propto w_max-w
    """
    translations = common.build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSoft',])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        common.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = locals()
        parameters.pop('self') 
        self.parameters = self.translate(parameters)


class SpikePairRule(common.SpikePairRule):
    
    translations = common.build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
    )
    possible_models = set(['StdwaSA','StdwaSoft'])
    
    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        common.SpikePairRule.__init__(self, tau_plus, tau_minus)
        parameters = locals()
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        