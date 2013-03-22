"""
Synapse Dynamics classes for the neuron module.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from pyNN.standardmodels import synapses, build_translations, STDPMechanism, SynapseDynamics

class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    
    __doc__ = synapses.TsodyksMarkramMechanism.__doc__    

    translations = build_translations(
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
        #synapses.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)
        self.parameters = self.translate({'U': U, 'tau_rec': tau_rec,
                                          'tau_facil': tau_facil, 'u0': u0,
                                          'x0': x0, 'y0': y0})

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    
    __doc__ = synapses.AdditiveWeightDependence.__doc__
    
    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSA',])
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        #synapses.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        self.parameters = self.translate({'w_min': w_min, 'w_max': w_max,
                                          'A_plus': A_plus, 'A_minus': A_minus})


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__ 

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaSoft',])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        #synapses.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        self.parameters = self.translate({'w_min': w_min, 'w_max': w_max,
                                          'A_plus': A_plus, 'A_minus': A_minus})

class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
    possible_models = set(['StdwaGuetig'])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        #synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self') 
        self.parameters = self.translate(parameters)
        self.parameters['muLTP'] = 0.0
        self.parameters['muLTD'] = 1.0


class GutigWeightDependence(synapses.GutigWeightDependence):
    
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
        ('mu_plus',   'muLTP'),
        ('mu_minus',  'muLTD'),
    )
    possible_models = set(['StdwaGuetig'])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01, mu_plus=0.5, mu_minus=0.5):
        #synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self') 
        self.parameters = self.translate(parameters)


class SpikePairRule(synapses.SpikePairRule):

    __doc__ = synapses.SpikePairRule.__doc__    

    translations = build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
    )
    possible_models = set(['StdwaSA', 'StdwaSoft', 'StdwaGuetig'])
    
    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        #synapses.SpikePairRule.__init__(self, tau_plus, tau_minus)
        self.parameters = self.translate({'tau_plus': tau_plus,
                                          'tau_minus': tau_minus})
        
