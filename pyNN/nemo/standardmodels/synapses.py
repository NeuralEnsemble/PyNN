"""
Synapse Dynamics classes for the nemo module.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: synapses.py 888 2011-01-04 15:17:54Z pierre $
"""
import numpy
from pyNN.standardmodels import synapses, SynapseDynamics, STDPMechanism

class STDPMechanism(STDPMechanism):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=0.0):
        assert dendritic_delay_fraction == 0, """Nemo does not currently support dendritic delays:
                                                 for the purpose of STDP calculations all delays
                                                 are assumed to be axonal."""
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction)

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    
    __doc__ = synapses.AdditiveWeightDependence.__doc__
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = parameters
        self.parameters['mu_plus']  = 0.
        self.parameters['mu_minus'] = 0.


class SpikePairRule(synapses.SpikePairRule):

    __doc__ = synapses.SpikePairRule.__doc__    

    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = parameters

    def pre_fire(self, precision=1.):
        return numpy.exp(-numpy.arange(0., 30, precision)/self.parameters['tau_plus'])

    def post_fire(self, precision=1.):
        return numpy.exp(-numpy.arange(0., 30, precision)/self.parameters['tau_minus'])

