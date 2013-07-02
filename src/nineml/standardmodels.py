# encoding: utf-8
"""
Standard cells for the nineml module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import nineml.user_layer as nineml

from pyNN.standardmodels import cells, synapses, electrodes, build_translations, StandardCurrentSource
from .simulator import state
from .utility import build_parameter_set, catalog_url, map_random_distribution_parameters


logger = logging.getLogger("PyNN")


class CellTypeMixin(object):
    
    @property
    def spiking_mechanism_parameters(self):
        smp = self.native_parameters
        for name in smp.keys():
            if name not in self.__class__.spiking_mechanism_parameter_names:
                smp.pop(name)
        return smp
    
    @property
    def synaptic_mechanism_parameters(self):
        smp = {}
        for receptor_type in self.__class__.receptor_types:
            smp[receptor_type] = {}
            for name in self.__class__.synaptic_mechanism_parameter_names[receptor_type]:
                smp[receptor_type][name.split("_")[1]] = self.native_parameters[name]
        return smp
    
    def to_nineml(self, label, shape):
        components = [self.spiking_node_to_nineml(label, shape)] + \
                     [self.synapse_type_to_nineml(st, label, shape) for st in self.receptor_types]
        return components
    
    def spiking_node_to_nineml(self, label, shape):
        return nineml.SpikingNodeType(
                    name="neuron type for population %s" % label,
                    definition=nineml.Definition(self.spiking_mechanism_definition_url),
                    parameters=build_parameter_set(self.spiking_mechanism_parameters, shape))
    
    def synapse_type_to_nineml(self, synapse_type, label, shape):
        return nineml.SynapseType(
                    name="%s post-synaptic response for %s" % (synapse_type, label),
                    definition=nineml.Definition(self.synaptic_mechanism_definition_urls[synapse_type]),
                    parameters=build_parameter_set(self.synaptic_mechanism_parameters[synapse_type], shape))


class IF_curr_exp(cells.IF_curr_exp, CellTypeMixin):
    
    __doc__ = cells.IF_curr_exp.__doc__      
    
    translations = build_translations(
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

    translations = build_translations(
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
   
    translations = build_translations(
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
    
    translations = build_translations(
        ('start',    'onset'),
        ('rate',     'frequency'),
        ('duration', 'duration'),
    )
    spiking_mechanism_definition_url = "%s/neurons/poisson_spike_source.xml" % catalog_url
    spiking_mechanism_parameter_names = ("onset", "frequency", "duration")


class SpikeSourceArray(cells.SpikeSourceArray, CellTypeMixin):

    __doc__ = cells.SpikeSourceArray.__doc__     
    
    translations = build_translations(
        ('spike_times',    'spike_times'),
    )
    spiking_mechanism_definition_url = "%s/neurons/spike_source_array.xml" % catalog_url
    spiking_mechanism_parameter_names = ("spike_times",)


class SynapseTypeMixin(object):
    counter = 0

    def to_nineml(self):
        return nineml.ConnectionType(
                            name="synapse type %d" % self.__class__.counter,
                            definition=nineml.Definition("%s/connectiontypes/%s" % (catalog_url, self.definition_file)),
                            parameters=build_parameter_set(self.parameters))
    

class StaticSynapse(SynapseTypeMixin, synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    definition_url = "%s/connectiontypes/static_connection.xml" % catalog_url

    translations = build_translations(
        ('weight', 'weight'),
        ('delay', 'delay'),
    )

    def _get_minimum_delay(self):
        return state.min_delay 


class CurrentSourceMixin(object):
    """Base class for a source of current to be injected into a neuron."""
    counter = 0
    
    def __init__(self):
        state.net.current_sources.append(self)
        self.__class__.counter += 1
        self.cell_list = []
    
    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        self.cell_list.extend(cell_list)
        
    def to_nineml(self):
        return nineml.CurrentSourceType(
                            name="current source %d" % self.__class__.counter,
                            definition=nineml.Definition("%s/currentsources/%s" % (catalog_url, self.definition_file)),
                            parameters=build_parameter_set(self.parameters))

class DCSource(CurrentSourceMixin, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )


class StepCurrentSource(CurrentSourceMixin, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )