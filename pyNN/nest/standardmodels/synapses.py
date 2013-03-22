"""
Synapse Dynamics classes for nest

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import nest
from pyNN.standardmodels import synapses, build_translations, SynapseDynamics, STDPMechanism

class SynapseDynamics(SynapseDynamics):
    __doc__ = SynapseDynamics.__doc__
    
    def _get_nest_synapse_model(self, suffix):
        # We create a particular synapse context for each projection, by copying
        # the one which is desired.
        if self.fast:
            if self.slow:
                raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
            else:
                base_model = self.fast.native_name
        elif self.slow:
            base_model = self.slow.possible_models
            if isinstance(base_model, set):
                logger.warning("Several STDP models are available for these connections:")
                logger.warning(", ".join(model for model in base_model))
                base_model = list(base_model)[0]
                logger.warning("By default, %s is used" % base_model)
        available_models = nest.Models(mtype='synapses') 
        if base_model not in available_models:
            raise ValueError("Synapse dynamics model '%s' not a valid NEST synapse model. "
                             "Possible models in your NEST build are: %s" % (base_model, available_models))
            
        synapse_defaults = nest.GetDefaults(base_model)
        for ignore in ('synapsemodel', 'num_connections', 'num_connectors',
                       'type', 'property_object'):
            synapse_defaults.pop(ignore, None)
        if self.fast:
            synapse_defaults.update(self.fast.parameters)
        elif self.slow:
            stdp_parameters = self.slow.all_parameters
            # NEST does not support w_min != 0
            try:
                stdp_parameters.pop("w_min_always_zero_in_NEST")
            except Exception:
                pass
            # Tau_minus is a parameter of the post-synaptic cell, not of the connection
            stdp_parameters.pop("tau_minus")
            synapse_defaults.update(stdp_parameters)                
        
        label = "%s_%s" % (base_model, suffix)
        nest.CopyModel(base_model, label, synapse_defaults)
        return label

    def _set_tau_minus(self, cells):
        if len(cells) > 0 and self.slow:
            if 'tau_minus' in nest.GetStatus([cells[0]])[0]:
                tau_minus = self.slow.timing_dependence.parameters["tau_minus"]
                nest.SetStatus(cells.tolist(), [{'tau_minus': tau_minus}])
            else:
                raise Exception("Postsynaptic cell model does not support STDP.")


class STDPMechanism(STDPMechanism):
    """Specification of STDP models."""
    
    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0):
        assert dendritic_delay_fraction == 1, """NEST does not currently support axonal delays:
                                                 for the purpose of STDP calculations all delays
                                                 are assumed to be dendritic."""
        super(STDPMechanism, self).__init__(timing_dependence, weight_dependence,
                                            voltage_dependence, dendritic_delay_fraction)


class TsodyksMarkramMechanism(synapses.TsodyksMarkramMechanism):
    
    __doc__ = synapses.TsodyksMarkramMechanism.__doc__    

    translations = build_translations(
        ('U', 'U'),
        ('tau_rec', 'tau_rec'),
        ('tau_facil', 'tau_fac'),
        ('u0', 'u'),  # this could cause problems for reverse translation
        ('x0', 'x' ), # (as for V_m) in cell models, since the initial value
        ('y0', 'y')   # is not stored, only set.
    )
    native_name = 'tsodyks_synapse'
    
    def __init__(self, U=0.5, tau_rec=100.0, tau_facil=0.0, u0=0.0, x0=1.0, y0=0.0):
        #synapses.TsodyksMarkramMechanism.__init__(self, U, tau_rec, tau_facil, u0, x0, y0)
        parameters = dict(locals()) # need the dict to get a copy of locals. When running
        parameters.pop('self')      # through coverage.py, for some reason, the pop() doesn't have any effect
        self.parameters = self.translate(parameters)

class AdditiveWeightDependence(synapses.AdditiveWeightDependence):

    __doc__ = synapses.AdditiveWeightDependence.__doc__     
    
    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01): # units?
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        #synapses.AdditiveWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
        self.parameters['mu_plus'] = 0.0
        self.parameters['mu_minus'] = 0.0


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__    

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        #synapses.MultiplicativeWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self') 
        self.parameters = self.translate(parameters)
        self.parameters['mu_plus'] = 1.0
        self.parameters['mu_minus'] = 1.0

class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__   

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        #synapses.AdditivePotentiationMultiplicativeDepression.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self') 
        self.parameters = self.translate(parameters)
        self.parameters['mu_plus'] = 0.0
        self.parameters['mu_minus'] = 1.0


class GutigWeightDependence(synapses.GutigWeightDependence):

    __doc__ = synapses.GutigWeightDependence.__doc__     

    translations = build_translations(
        ('w_max',     'Wmax',  1000.0), # unit conversion
        ('w_min',     'w_min_always_zero_in_NEST'),
        ('A_plus',    'lambda'),
        ('A_minus',   'alpha', 'A_minus/A_plus', 'alpha*lambda'),
        ('mu_plus',   'mu_plus'),
        ('mu_minus',  'mu_minus'),
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
        
    def __init__(self, w_min=0.0, w_max=1.0, A_plus=0.01, A_minus=0.01,mu_plus=0.5,mu_minus=0.5):
        if w_min != 0:
            raise Exception("Non-zero minimum weight is not supported by NEST.")
        #synapses.GutigWeightDependence.__init__(self, w_min, w_max, A_plus, A_minus)
        parameters = dict(locals())
        parameters.pop('self') 
        self.parameters = self.translate(parameters)


class SpikePairRule(synapses.SpikePairRule):
    
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tau_plus'),
        ('tau_minus', 'tau_minus'), # defined in post-synaptic neuron
    )
    possible_models = set(['stdp_synapse']) #,'stdp_synapse_hom'])
    
    def __init__(self, tau_plus=20.0, tau_minus=20.0):
        #synapses.SpikePairRule.__init__(self, tau_plus, tau_minus)
        parameters = dict(locals())
        parameters.pop('self')
        self.parameters = self.translate(parameters)
