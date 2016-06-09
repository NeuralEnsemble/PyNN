# encoding: utf-8
"""
Standard cells for the nineml module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import nineml.user as nineml
import nineml.abstraction as al

from pyNN.standardmodels import cells, synapses, electrodes, build_translations, StandardCurrentSource
from .simulator import state
from .utility import (build_parameter_set, catalog_url,
                      map_random_distribution_parameters)


logger = logging.getLogger("PyNN")


class CellTypeMixin(object):

    @property
    def spiking_component_parameters(self):
        smp = self.native_parameters
        for name in smp.keys():
            if name not in self.__class__.spiking_component_parameter_names:
                smp.pop(name)
        return smp

    @property
    def synaptic_receptor_parameters(self):
        smp = {}
        for receptor_type in self.__class__.receptor_types:
            smp[receptor_type] = {}
            for name in self.__class__.synaptic_receptor_parameter_names[receptor_type]:
                smp[receptor_type][name.split(".")[1]] = self.native_parameters[name]
        return smp

    def to_nineml(self, label, shape):
        inline_definition = False  # todo: get inline definitions to work
        components = [self.spiking_component_to_nineml(label, shape, inline_definition)] + \
                     [self.synaptic_receptor_component_to_nineml(st, label, shape, inline_definition) for st in self.receptor_types]
        return components

    def spiking_component_to_nineml(self, label, shape, inline_definition=False):
        """
        Return a 9ML user layer Component describing the neuron type.
        (mathematical model + parameterization).
        """
        if inline_definition:
            definition = self.spiking_component_type_to_nineml()
        else:
            definition = self.spiking_component_definition_url
        return nineml.SpikingNodeType(
                    name="neuron type for population %s" % label,
                    definition=nineml.Definition(definition, "dynamics"),
                    parameters=build_parameter_set(self.spiking_component_parameters, shape))

    def synaptic_receptor_component_to_nineml(self, receptor_type, label, shape, inline_definition=False):
        """
        Return a 9ML user layer Component describing the post-synaptic mechanism
        (mathematical model + parameterization).

        Note that we use the name "receptor" as a shorthand for "receptor,
        ion channel and any associated signalling mechanisms".
        """
        if inline_definition:
            definition = self.synaptic_receptor_component_type_to_nineml(receptor_type)
        else:
            definition = self.synaptic_receptor_component_definition_urls[receptor_type]
        return nineml.SynapseType(
                    name="%s post-synaptic response for %s" % (receptor_type, label),
                    definition=nineml.Definition(definition, "dynamics"),
                    parameters=build_parameter_set(self.synaptic_receptor_parameters[receptor_type], shape))


#class IF_curr_exp(cells.IF_curr_exp, CellTypeMixin):
#    
#    __doc__ = cells.IF_curr_exp.__doc__      
#    
#    translations = build_translations(
#        ('tau_m',      'membraneTimeConstant'),
#        ('cm',         'membraneCapacitance'),
#        ('v_rest',     'restingPotential'),
#        ('v_thresh',   'threshold'),
#        ('v_reset',    'resetPotential'),
#        ('tau_refrac', 'refractoryTime'),
#        ('i_offset',   'offsetCurrent'),
#        ('tau_syn_E',  'excitatory_decayTimeConstant'),
#        ('tau_syn_I',  'inhibitory_decayTimeConstant'),
#    )
#    spiking_component_definition_url = "%s/neurons/IaF_tau.xml" % catalog_url
#    synaptic_component_definition_urls = {
#        'excitatory': "%s/postsynapticresponses/exp_i.xml" % catalog_url,
#        'inhibitory': "%s/postsynapticresponses/exp_i.xml" % catalog_url
#    }
#    spiking_component_parameter_names = ('membraneTimeConstant','membraneCapacitance',
#                                         'restingPotential', 'threshold',
#                                         'resetPotential', 'refractoryTime')
#    synaptic_component_parameter_names = {
#        'excitatory': ['excitatory_decayTimeConstant',],
#        'inhibitory': ['inhibitory_decayTimeConstant',]
#    }


class IF_cond_exp(cells.IF_cond_exp, CellTypeMixin):

    __doc__ = cells.IF_cond_exp.__doc__    

    translations = build_translations(
        ('tau_m',      'tau_m'),
        ('cm',         'cm'),
        ('v_rest',     'v_rest'),
        ('v_thresh',   'v_thresh'),
        ('v_reset',    'v_reset'),
        ('tau_refrac', 'tau_refrac'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'excitatory.tau_syn'),
        ('tau_syn_I',  'inhibitory.tau_syn'),
        ('e_rev_E',    'excitatory.e_rev'),
        ('e_rev_I',    'inhibitory.e_rev')
    )
    spiking_component_definition_url = "%s/neurons/iaf_tau.xml" % catalog_url
    synaptic_receptor_component_definition_urls = {
        'excitatory': "%s/postsynapticresponses/cond_exp_syn.xml" % catalog_url,
        'inhibitory': "%s/postsynapticresponses/cond_exp_syn.xml" % catalog_url
    }
    spiking_component_parameter_names = ('tau_m', 'cm',
                                         'v_rest', 'v_thresh',
                                         'v_reset', 'i_offset',
                                         'tau_refrac')
    synaptic_receptor_parameter_names = {
        'excitatory': ['excitatory.tau_syn', 'excitatory.e_rev'],
        'inhibitory': ['inhibitory.tau_syn', 'inhibitory.e_rev']
    }

    @classmethod
    def spiking_component_type_to_nineml(cls):
        """Return a 9ML ComponentClass describing the neuron model."""
        iaf = al.ComponentClass(
            name="iaf_tau",
            regimes=[
                al.Regime(
                    name="subthreshold_regime",
                    time_derivatives=["dv/dt = (v_rest - v)/tau_m + (i_offset + i_syn)/cm"],
                    transitions=al.On("v > v_thresh",
                                      do=["t_spike = t",
                                          "v = v_reset",
                                          al.OutputEvent('spike_output')],
                                      to="refractory_regime"),
                ),  
                al.Regime(
                    name="refractory_regime",
                    time_derivatives=["dv/dt = 0"],
                    transitions=[al.On("t >= t_spike + tau_refrac",
                                       to="subthreshold_regime")],
                )
            ],
            state_variables=[
                al.StateVariable('v'), #, dimension='[V]' # '[V]' should be an alias for [M][L]^2[T]^-3[I]^-1
                al.StateVariable('t_spike'), #, dimension='[T]'
            ],
            analog_ports=[al.AnalogSendPort("v"),
                          al.AnalogReducePort("i_syn", reduce_op="+"), ],
            event_ports=[al.EventSendPort('spike_output'), ],
            parameters=['cm', 'tau_refrac', 'tau_m', 'v_reset', 'v_rest', 'v_thresh', 'i_offset']  # add dimensions, or infer them from dimensions of variables
                                                                                                   # in fact, we should be able to infer what are the parameters, without listing them
        )
        return iaf

    @classmethod
    def synaptic_receptor_component_type_to_nineml(cls, synapse_type):
        """Return a 9ML ComponentClass describing the synaptic receptor model."""
        coba = al.ComponentClass(
            name="cond_exp_syn",
            aliases=["i_syn:=g_syn*(e_rev-v)", ],
            regimes=[
                al.Regime(
                    name="coba_default_regime",
                    time_derivatives=["dg_syn/dt = -g_syn/tau_syn", ],
                    transitions=al.On('spike_input', do=["g_syn=g_syn+q"]),
                )
            ],
            state_variables=[al.StateVariable('g_syn')],  #, dimension='[G]'  # alias [M]^-1[L]^-2[T]^3[I]^2
            analog_ports=[al.AnalogReceivePort("v"), al.AnalogSendPort("i_syn"), al.AnalogReceivePort('q')],
            parameters=['tau_syn', 'e_rev']
        )
        return coba

        #iaf_2coba_model = al.ComponentClass(
        #name="IF_cond_exp",
        #subnodes={"cell": iaf,
        #          "excitatory": coba,
        #          "inhibitory": coba})
        #iaf_2coba_model.connect_ports("cell.v", "excitatory.v")
        #iaf_2coba_model.connect_ports("cell.v", "inhibitory.v")
        #iaf_2coba_model.connect_ports("excitatory.i_syn", "cell.i_syn")
        #iaf_2coba_model.connect_ports("excitatory.i_syn", "cell.i_syn")
        #
        #return iaf_2coba_model


#class IF_cond_alpha(cells.IF_cond_exp, CellTypeMixin):
#
#    __doc__ = cells.IF_cond_alpha.__doc__    
#   
#    translations = build_translations(
#        ('tau_m',      'membraneTimeConstant'),
#        ('cm',         'membraneCapacitance'),
#        ('v_rest',     'restingPotential'),
#        ('v_thresh',   'threshold'),
#        ('v_reset',    'resetPotential'),
#        ('tau_refrac', 'refractoryTime'),
#        ('i_offset',   'offsetCurrent'),
#        ('tau_syn_E',  'excitatory_timeConstant'),
#        ('tau_syn_I',  'inhibitory_timeConstant'),
#        ('e_rev_E',    'excitatory_reversalPotential'),
#        ('e_rev_I',    'inhibitory_reversalPotential')
#    )
#    spiking_component_definition_url = "%s/neurons/IaF_tau.xml" % catalog_url
#    synaptic_component_definition_urls = {
#        'excitatory': "%s/postsynapticresponses/alpha_g.xml" % catalog_url,
#        'inhibitory': "%s/postsynapticresponses/alpha_g.xml" % catalog_url
#    }
#    spiking_component_parameter_names = ('membraneTimeConstant','membraneCapacitance',
#                                         'restingPotential', 'threshold',
#                                         'resetPotential', 'refractoryTime')
#    synaptic_component_parameter_names = {
#        'excitatory': ['excitatory_timeConstant', 'excitatory_reversalPotential'],
#        'inhibitory': ['inhibitory_timeConstant',  'inhibitory_reversalPotential']
#    }


class SpikeSourcePoisson(cells.SpikeSourcePoisson, CellTypeMixin):

    __doc__ = cells.SpikeSourcePoisson.__doc__     

    translations = build_translations(
        ('start',    'start'),
        ('rate',     'rate'),
        ('duration', 'duration'),
    )
    spiking_component_definition_url = "%s/neurons/poisson_spike_source.xml" % catalog_url
    spiking_component_parameter_names = ("onset", "frequency", "duration")

    @classmethod
    def spiking_component_type_to_nineml(cls):
        """Return a 9ML ComponentClass describing the spike source model."""
        source = al.ComponentClass(
            name="poisson_spike_source",
            regimes=[
                al.Regime(
                    name="before",
                    transitions=[al.On("t > start",
                                       do=["t_spike = -1"],
                                       to="on")]),
                al.Regime(
                    name="on",
                    transitions=[al.On("t >= t_spike",
                                       do=["t_spike = t_spike + random.exponential(rate)",
                                           al.OutputEvent('spike_output')]),
                                 al.On("t >= start + duration",
                                       to="after")],
                ),
                al.Regime(name="after")
            ],
            state_variables=[
                al.StateVariable('t_spike'), #, dimension='[T]'
            ],
            event_ports=[al.EventSendPort('spike_output'), ],
            parameters=['start', 'rate', 'duration'],  # add dimensions, or infer them from dimensions of variables
        )
        return source


class SpikeSourceArray(cells.SpikeSourceArray, CellTypeMixin):

    __doc__ = cells.SpikeSourceArray.__doc__     

    translations = build_translations(
        ('spike_times',    'spike_times'),
    )
    spiking_component_definition_url = "%s/neurons/spike_source_array.xml" % catalog_url
    spiking_component_parameter_names = ("spike_times",)

    @classmethod
    def spiking_component_type_to_nineml(cls):
        """Return a 9ML ComponentClass describing the spike source model."""
        source = al.ComponentClass(
            name="spike_source_array",
            regimes=[
                al.Regime(
                    name="on",
                    transitions=[al.On("t >= spike_times[i]",  # this is currently illegal
                                       do=["i = i + 1",
                                           al.OutputEvent('spike_output')])],
                ),
            ],
            state_variables=[
                al.StateVariable('t_spike'), #, dimension='[T]'
                al.StateVariable('i'), #, dimension='[T]'
            ],
            event_ports=[al.EventSendPort('spike_output'), ],
            parameters=['start', 'rate', 'duration'],  # add dimensions, or infer them from dimensions of variables
        )
        return source


class SynapseTypeMixin(object):
    counter = 0

    def to_nineml(self):
        return nineml.ConnectionType(
                            name="synapse type %d" % self.__class__.counter,
                            definition=nineml.Definition("%s/connectiontypes/%s" % (catalog_url, self.definition_file),
                                                         "dynamics"),
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
                            definition=nineml.Definition("%s/currentsources/%s" % (catalog_url, self.definition_file),
                                                         "dynamics"),
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
