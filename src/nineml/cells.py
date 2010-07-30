"""
Standard cells for 9ML
"""

from pyNN import standardmodels, cells
import nineml.user_layer as nineml
from utility import build_parameter_set, catalog_url, map_random_distribution_parameters


class CellTypeMixin(object):
    
    @property
    def spiking_mechanism_parameters(self):
        smp = {}
        for name in self.__class__.spiking_mechanism_parameter_names:
            smp[name] = self.parameters[name]
        return smp
    
    @property
    def synaptic_mechanism_parameters(self):
        smp = {}
        for synapse_type in self.__class__.synapse_types:
            smp[synapse_type] = {}
            for name in self.__class__.synaptic_mechanism_parameter_names[synapse_type]:
                smp[synapse_type][name.split("_")[1]] = self.parameters[name]
        return smp
    
    def to_nineml(self, label):
        components = []
        component = nineml.SpikingNodeType(
                                    name="%s neuron type" % label,
                                    definition=nineml.Definition(self.spiking_mechanism_definition_url),
                                    parameters=build_parameter_set(self.spiking_mechanism_parameters))
        components.append(component)             
        for synapse_type in self.synapse_types:
            component = nineml.SynapseType(
                                    name="%s %s post-synaptic response" % (label, synapse_type),
                                    definition=nineml.Definition(self.synaptic_mechanism_definition_urls[synapse_type]),
                                    parameters=build_parameter_set(self.synaptic_mechanism_parameters[synapse_type]))
            components.append(component)
        return components


class IF_curr_exp(cells.IF_curr_exp):
    """Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses."""
    
    translations = standardmodels.build_translations(
        ('tau_m',      'membraneTimeConstant'),
        ('cm',         'membraneCapacitance'),
        ('v_rest',     'restingPotential'),
        ('v_thresh',   'threshold'),
        ('v_reset',    'resetPotential'),
        ('tau_refrac', 'refractoryTime'),
        ('i_offset',   'offsetCurrent'),
        ('tau_syn_E',  'excitatory_decayTimeConstant'),
        ('tau_syn_I',  'inhibitory_decayTimeConstant'),
    )
    spiking_mechanism_definition_url = "%s/neurons/IaF_tau.xml" % catalog_url
    synaptic_mechanism_definition_urls = {
        'excitatory': "%s/postsynapticresponses/exp_i.xml" % catalog_url,
        'inhibitory': "%s/postsynapticresponses/exp_i.xml" % catalog_url
    }
    spiking_mechanism_parameter_names = ('membraneTimeConstant','membraneCapacitance',
                                         'restingPotential', 'threshold',
                                         'resetPotential', 'refractoryTime')
    synaptic_mechanism_parameter_names = {
        'excitatory': ['excitatory_decayTimeConstant',],
        'inhibitory': ['inhibitory_decayTimeConstant',]
    }


class IF_cond_exp(cells.IF_cond_exp, CellTypeMixin):
   
    translations = standardmodels.build_translations(
        ('tau_m',      'membraneTimeConstant'),
        ('cm',         'membraneCapacitance'),
        ('v_rest',     'restingPotential'),
        ('v_thresh',   'threshold'),
        ('v_reset',    'resetPotential'),
        ('tau_refrac', 'refractoryTime'),
        ('i_offset',   'offsetCurrent'),
        ('tau_syn_E',  'excitatory_decayTimeConstant'),
        ('tau_syn_I',  'inhibitory_decayTimeConstant'),
        ('e_rev_E',    'excitatory_reversalPotential'),
        ('e_rev_I',    'inhibitory_reversalPotential')
    )
    spiking_mechanism_definition_url = "%s/neurons/IaF_tau.xml" % catalog_url
    synaptic_mechanism_definition_urls = {
        'excitatory': "%s/postsynapticresponses/exp_g.xml" % catalog_url,
        'inhibitory': "%s/postsynapticresponses/exp_g.xml" % catalog_url
    }
    spiking_mechanism_parameter_names = ('membraneTimeConstant','membraneCapacitance',
                                         'restingPotential', 'threshold',
                                         'resetPotential', 'refractoryTime')
    synaptic_mechanism_parameter_names = {
        'excitatory': ['excitatory_decayTimeConstant', 'excitatory_reversalPotential'],
        'inhibitory': ['inhibitory_decayTimeConstant',  'inhibitory_reversalPotential']
    }


class IF_cond_alpha(cells.IF_cond_exp, CellTypeMixin):
   
    translations = standardmodels.build_translations(
        ('tau_m',      'membraneTimeConstant'),
        ('cm',         'membraneCapacitance'),
        ('v_rest',     'restingPotential'),
        ('v_thresh',   'threshold'),
        ('v_reset',    'resetPotential'),
        ('tau_refrac', 'refractoryTime'),
        ('i_offset',   'offsetCurrent'),
        ('tau_syn_E',  'excitatory_timeConstant'),
        ('tau_syn_I',  'inhibitory_timeConstant'),
        ('e_rev_E',    'excitatory_reversalPotential'),
        ('e_rev_I',    'inhibitory_reversalPotential')
    )
    spiking_mechanism_definition_url = "%s/neurons/IaF_tau.xml" % catalog_url
    synaptic_mechanism_definition_urls = {
        'excitatory': "%s/postsynapticresponses/alpha_g.xml" % catalog_url,
        'inhibitory': "%s/postsynapticresponses/alpha_g.xml" % catalog_url
    }
    spiking_mechanism_parameter_names = ('membraneTimeConstant','membraneCapacitance',
                                         'restingPotential', 'threshold',
                                         'resetPotential', 'refractoryTime')
    synaptic_mechanism_parameter_names = {
        'excitatory': ['excitatory_timeConstant', 'excitatory_reversalPotential'],
        'inhibitory': ['inhibitory_timeConstant',  'inhibitory_reversalPotential']
    }
    

class SpikeSourcePoisson(cells.SpikeSourcePoisson, CellTypeMixin):
    
    translations = standardmodels.build_translations(
        ('start',    'onset'),
        ('rate',     'frequency'),
        ('duration', 'duration'),
    )
    spiking_mechanism_definition_url = "%s/neurons/poisson_spike_source.xml" % catalog_url
    spiking_mechanism_parameter_names = ("onset", "frequency", "duration")
