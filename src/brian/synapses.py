"""
Synapse Dynamics classes for the brian module.

$Id$
"""

from pyNN import common


class SynapseDynamics(common.SynapseDynamics):
    def __init__(self, fast=None, slow=None):
        synapses.SynapseDynamics.__init__(self, fast, slow)

class STDPMechanism(common.STDPMechanism):
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        assert dendritic_delay_fraction == 1, """Brian does not currently support axonal delays:
                                                 for the purpose of STDP calculations all delays
                                                 are assumed to be dendritic."""
        synapses.STDPMechanism.__init__(self, timing_dependence, weight_dependence,
                                      voltage_dependence, dendritic_delay_fraction)

class TsodkysMarkramMechanism(common.ModelNotAvailable):
    
    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        synapses.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)
        self.parameters = self.translate(parameters)
        self.eqs = '''
              dR/dt=(1-R)/%g : 1
              tau_rec        : ms
              ''' %tau_rec

    def reset(population,spikes, v_reset):
        population.R_[spikes]-=U_SE*population.R_[spikes]
        population.v_[spikes]= v_reset

class AdditiveWeightDependence(common.ModelNotAvailable):
    pass

class MultiplicativeWeightDependence(common.ModelNotAvailable):
    pass

class SpikePairRule(common.ModelNotAvailable):
    pass
