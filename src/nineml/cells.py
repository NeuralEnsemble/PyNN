"""
Standard cells for 9ML

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from __future__ import absolute_import
from pyNN import standardmodels
import pyNN.standardmodels.cells as cells
import nineml.user_layer as nineml
from pyNN.nineml.utility import build_parameter_set, catalog_url, map_random_distribution_parameters

from pyNN.models import BaseCellType
import nineml.abstraction_layer as nineml
import logging
import os
import re
from itertools import chain

logger = logging.getLogger("PyNN")


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
        components = [self.spiking_node_to_nineml(label)] + \
                     [self.synapse_type_to_nineml(st, label) for st in self.synapse_types]
        return components
    
    def spiking_node_to_nineml(self, label):
        return nineml.SpikingNodeType(
                    name="neuron type for population %s" % label,
                    definition=nineml.Definition(self.spiking_mechanism_definition_url),
                    parameters=build_parameter_set(self.spiking_mechanism_parameters))
    
    def synapse_type_to_nineml(self, synapse_type, label):
        return nineml.SynapseType(
                    name="%s post-synaptic response for %s" % (synapse_type, label),
                    definition=nineml.Definition(self.synaptic_mechanism_definition_urls[synapse_type]),
                    parameters=build_parameter_set(self.synaptic_mechanism_parameters[synapse_type]))


class IF_curr_exp(cells.IF_curr_exp, CellTypeMixin):
    
    __doc__ = cells.IF_curr_exp.__doc__      
    
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
   
    __doc__ = cells.IF_cond_exp.__doc__    

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

    __doc__ = cells.IF_cond_alpha.__doc__    
   
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

    __doc__ = cells.SpikeSourcePoisson.__doc__     
    
    translations = standardmodels.build_translations(
        ('start',    'onset'),
        ('rate',     'frequency'),
        ('duration', 'duration'),
    )
    spiking_mechanism_definition_url = "%s/neurons/poisson_spike_source.xml" % catalog_url
    spiking_mechanism_parameter_names = ("onset", "frequency", "duration")



# Neuron Models derived from a 9ML AL definition

class NineMLCellType(BaseCellType):
    #model = NineMLCell
    
    def __init__(self, parameters):
        BaseCellType.__init__(self, parameters)
        self.parameters["type"] = self


def unimplemented_builder(*args, **kwargs):
    raise NotImplementedError, "TODO: 9ML neuron builder"

def nineml_cell_type(name, neuron_model, port_map={}, weight_variables={}, **synapse_models):
    """
    Return a new NineMLCellType subclass.
    """
    return _build_nineml_celltype(name, (NineMLCellType,),
                                  {'neuron_model': neuron_model,
                                   'synapse_models': synapse_models,
                                   'port_map': port_map,
                                   'weight_variables': weight_variables,
                                   'builder': unimplemented_builder})

# Helpers for Neuron Models derived from a 9ML AL definition


def _add_prefix(synapse_model, prefix, port_map):
    assert False, "Deprecated" 
    """
    Add a prefix to all variables in `synapse_model`, except for variables with
    receive ports and specified in `port_map`.
    """
    synapse_model.__cache__ = {}
    exclude = []
    new_port_map = []
    for name1, name2 in port_map:
        if synapse_model.ports_map[name2].mode == 'recv':
            exclude.append(name2)
            new_port_map.append((name1, name2))
        else:
            new_port_map.append((name1, prefix + '_' + name2))
    synapse_model.add_prefix(prefix + '_', exclude=exclude)
    return new_port_map





class _mh_build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    Called by nineml_celltype_from_model
    """
    def __new__(cls, name, bases, dct):
        
        import nineml.abstraction_layer as al
        from nineml.abstraction_layer import flattening, writers, component_modifiers

        #Extract Parameters Back out from Dict:
        combined_model = dct['nineml_model']
        synapse_components = dct['synapse_components']

        # Flatten the model:
        assert isinstance(combined_model, al.ComponentClass)
        if combined_model.is_flat():
            flat_component = combined_model
        else:
            flat_component = flattening.flatten( combined_model,name )
        
        # Make the substitutions:
        flat_component.backsub_all()
        #flat_component.backsub_aliases()
        #flat_component.backsub_equations()

        # Close any open reduce ports:
        component_modifiers.ComponentModifier.close_all_reduce_ports(component = flat_component)
         
        # New:
        dct["combined_model"] = flat_component
        dct["default_parameters"] = dict( (param.name, 1.0) for param in flat_component.parameters )
        dct["default_initial_values"] = dict((statevar.name, 0.0) for statevar in chain(flat_component.state_variables) )
        dct["synapse_types"] = [syn.namespace for syn in synapse_components] 
        dct["standard_receptor_type"] = (dct["synapse_types"] == ('excitatory', 'inhibitory'))
        dct["injectable"] = True # need to determine this. How??
        dct["conductance_based"] = True # how to determine this??
        dct["model_name"] = name
        
        
        # Recording from bindings:
        dct["recordable"] = [port.name for port in flat_component.analog_ports] + ['spikes', 'regime'] + [alias.lhs for alias in flat_component.aliases] + [statevar.name for statevar in flat_component.state_variables]
        
        dct["weight_variables"] = dict([ (syn.namespace,syn.namespace+'_'+syn.weight_connector )
                                         for syn in synapse_components ])
        
        
        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        # generate and compile NMODL code, then load the mechanism into NEUORN
        dct["builder"](flat_component, dct["weight_variables"], hierarchical_mode=True)
        
        return type.__new__(cls, name, bases, dct)
        

      
class CoBaSyn(object):
    def __init__(self, namespace, weight_connector):
        self.namespace = namespace
        self.weight_connector = weight_connector





