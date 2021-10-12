# encoding: utf-8
"""
Definition of cell classes for the neuron module.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
from math import pi
from collections import defaultdict
from functools import reduce
import numpy as np
#from neuron import h, nrn, hclass
import arbor
import re
import numpy.random

from pyNN import errors
#from pyNN.models import BaseCellType
from pyNN.morphology import NeuriteDistribution, IonChannelDistribution
from .standardmodels.ion_channels import LeakyChannel
from .recording import recordable_pattern
from .simulator import state

logger = logging.getLogger("PyNN")

class ArborTemplate(object):

    def __init__(self, morphology, cm, Ra, ionic_species, **other_parameters):
        import neuroml
        import neuroml.arraymorph

        self.__decor_class__ = decor()

        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None
        self.spike_times = h.Vector(0)# change this for arbor

        # create morphology
        self.morphology = morphology
        self.ionic_species = ionic_species
        #self.sections = {} # is this needed for arbor?
        #self.section_labels = {} # dito

        # create morphology
        self.sections = self.create_arbor_tree() # arbor segment tree ~ NEURON section i.e unbranched collection of segments
        self.section_labels = arbor.label_dict(self.create_region_definitions())
        # create cell
        self.cell = arbor.cable_cell(self.sections, self.section_labels)
        # set cell properties
        self.cell.set_properties(cm=cm, rL=Ra)
        # setup ion, setup channel mechanism and attach the mechanism
        self.setup_ionic_species()
        self.setup_channel_mechanism(other_parameters)
        self.attach_mechanisms(other_parameters)

        for receptor_name in self.post_synaptic_entities:
            self.morphology.synaptic_receptors[receptor_name] = defaultdict(list)

        # if isinstance(morphology._morphology, neuroml.arraymorph.ArrayMorphology):
        #     M = morphology._morphology
        #     for i in range(len(morphology._morphology)):
        #         vertex = M.vertices[i]
        #         parent_index = M.connectivity[i]
        #         parent = M.vertices[parent_index]
        #         section = nrn.Section()
        #         for v in (vertex, parent):
        #             x, y, z, d = v
        #             h.pt3dadd(x, y, z, d, sec=section)
        #         section.nseg = 1
        #         section.cm = cm
        #         section.Ra = Ra
        #         # ignore fractions_along for now
        #         if i > 1:
        #             section.connect(self.sections[parent_index], DISTAL, PROXIMAL)
        #         self.sections[i] = section
        #     self.morphology._soma_index = 0  # fragile temporary hack - should be index of the vertex with no parent
        # elif isinstance(morphology._morphology, neuroml.Morphology):
        #     unresolved_connections = []
        #     for index, segment in enumerate(morphology.segments):
        #         section = nrn.Section(name=segment.name)
        #         section.L = segment.length
        #         section(PROXIMAL).diam = segment.proximal.diameter
        #         section(DISTAL).diam = segment.distal.diameter
        #         section.nseg = 1
        #         if isinstance(cm, NeuriteDistribution):
        #             section.cm = cm.value_in(self.morphology, index)
        #         else:
        #             section.cm = cm
        #         section.Ra = Ra
        #         segment_id = segment.id
        #         assert segment_id is not None
        #         if segment.parent:
        #             parent_id = segment.parent.id
        #             connection_point = DISTAL  # should generalize
        #             if segment.parent.id in self.sections:
        #                 section.connect(self.sections[parent_id], connection_point, PROXIMAL)
        #             else:
        #                 unresolved_connections.append((segment_id, parent_id))
        #         self.sections[segment_id] = section
        #         if segment.name == "soma":
        #             self.morphology._soma_index = segment_id
        #         if segment.name is not None:
        #             self.section_labels[segment.name] = section
        #         segment._section = section
        #     for section_id, parent_id in unresolved_connections:
        #         self.sections[section_id].connect(self.sections[parent_id], DISTAL, PROXIMAL)
        # else:
        #     raise ValueError("{} not supported as a neuron morphology".format(type(morphology)))

        # insert ion channels
        # for name, ion_channel in self.ion_channels.items():
        #     parameters = other_parameters[name]
        #     mechanism_name = ion_channel.model
        #     conductance_density = parameters[ion_channel.conductance_density_parameter]
        #     for index, id in enumerate(self.sections):
        #         g = conductance_density.value_in(self.morphology, index)
        #         if g is not None and g > 0:
        #             section = self.sections[id]
        #             section.insert(mechanism_name)
        #             varname = ion_channel.conductance_density_parameter + "_" + ion_channel.model
        #             setattr(section, varname, g)
        #             ##print(index, mechanism_name, ion_channel.conductance_density_parameter, g)
        #             # temporary hack - we're not using the leak conductance from the hh mechanism,
        #             # so set the conductance to zero
        #             if mechanism_name == "hh":
        #                 setattr(section, "gl_hh", 0.0)
        #             for param_name, value in parameters.items():
        #                 if param_name != ion_channel.conductance_density_parameter:
        #                     if isinstance(value, IonChannelDistribution):
        #                         value = value.value_in(self.morphology, index)
        #                     try:
        #                         setattr(section, param_name + "_" + ion_channel.model, value)
        #                     except AttributeError:  # e.g. parameters not defined within a mechanism, e.g. ena
        #                         setattr(section, param_name, value)
        #                     ##print(index, mechanism_name, param_name, value)

        # insert post-synaptic mechanisms
        for name, pse in self.post_synaptic_entities.items():
            parameters = other_parameters[name]
            mechanism_name = pse.model
            synapse_model = getattr(h, mechanism_name)
            density_function = parameters["density"]
            for index, id in enumerate(self.sections):
                density = density_function.value_in(self.morphology, index)
                if density > 0:
                    n_synapses, remainder = divmod(density, 1)
                    rnd = numpy.random  # todo: use the RNG from the parent Population
                    if rnd.uniform() < remainder:
                        n_synapses += 1
                    section = self.sections[id]
                    for i in range(int(n_synapses)):
                        self.morphology.synaptic_receptors[name][id].append(synapse_model(0.5, sec=section))

        # handle ionic species
        def set_in_section(section, index, name, value):
            if isinstance(value, IonChannelDistribution):  # should be "NeuriteDistribution"
                value = value.value_in(self.morphology, index)
            if value is not None:
                if name == "eca":     # tmp hack
                    section.push()
                    h.ion_style("ca_ion", 1, 1, 0, 1, 0)
                    h.pop_section()
                try:
                    setattr(section, name, value)
                except (NameError, AttributeError) as err:  # section does not contain ion
                    if "the mechanism does not exist" not in str(err):
                        raise

        for ion_name, parameters in self.ionic_species.items():
            for index, id in enumerate(self.sections):
                section = self.sections[id]
                set_in_section(section, index, "e{}".format(ion_name), parameters.reversal_potential)
                if parameters.internal_concentration:
                    set_in_section(section, index, "{}i".format(ion_name), parameters.internal_concentration)
                if parameters.external_concentration:
                    set_in_section(section, index, "{}o".format(ion_name), parameters.external_concentration)


        # set source section
        if self.spike_source:
            self.source_section = self.sections[self.spike_source]
        elif "axon_initial_segment" in self.sections:
            self.source_section = self.sections["axon_initial_segment"]
        else:
            self.source_section = self.sections[morphology.soma_index]
        self.source = self.source_section(0.5)._ref_v
        self.rec = h.NetCon(self.source, None, sec=self.source_section)

    ######################################
    # Functions to create Arbor morphology
    ######################################
    def create_arbor_tree(self):
        tree = arbor.segment_tree()
        for i, nml_seg in enumerate(self.morphology.segments):
            self.append_arbor_tree(tree, nml_seg)
        return tree # arbor segment tree ~ NEURON section i.e unbranched collection of segments

    def append_arbor_tree(self, tree, nml_seg):
        if not nml_seg.parent:
            tree.append(arbor.mnpos,
                        arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                     nml_seg.proximal.diameter / 2),
                        arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                     nml_seg.distal.diameter / 2), tag=self.get_swc_tag(nml_seg))
        else:
            tree.append(nml_seg.parent.id,
                        arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                     nml_seg.proximal.diameter / 2),
                        arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                     nml_seg.distal.diameter / 2), tag=self.get_swc_tag(nml_seg))

    # Arbor uses a domains specific language (DSL) to describe regions and locations, which are given labels.
    def create_region_definitions(self):
        dict_defs = {}
        for i, nml_seg in enumerate(self.morphology.segments):
            dict_defs.update({nml_seg.name: "(tag " + str(self.get_swc_tag(nml_seg)) + ")"})
        dict_defs.update({"everywhere": "(all)"})
        return dict_defs

    def get_swc_tag(self, nml_seg):
        if re.search("soma", nml_seg.name, re.IGNORECASE):
            return 1
        elif re.search("axon", nml_seg.name, re.IGNORECASE):
            return 2
        elif re.search("dend", nml_seg.name, re.IGNORECASE):
            return 3
        else:
            return 5
    ######################################

    ######################################
    # Functions to set ions in the cell
    ######################################
    def setup_ionic_species(self):
        for ion_name, ion_sp in self.ionic_species.items():
            self.cell.set_ion(ion=ion_name, rev_pot=ion_sp.reversal_potential)
    ######################################

    #self.setup_channel_mechanism(other_parameters) # returns self.mechanisms
    ######################################
    # Functions to set channel mechanisms
    ######################################
    def setup_channel_mechanisms(self, other_parameters):
        self.mechanisms = {}
        for channel_name, ion_channel in self.ion_channels.items():
            mechanism_name = ion_channel.model
            its_param_dict = self.mechanism_parameters_dict(mechanism_name, other_parameters)
            self.mechanisms.update(
                {mechanism_name: arbor.mechanism(mechanism_name, its_param_dict)})

    def mechanism_parameters_dict(self, mechanism_name, other_parameters):
        get_dict_for_pas = lambda chnnl_type, param_key, param_value:\
            {chnnl_type.conductance_density_parameter: param_value.value} if (param_key == "conductance_density")\
                else {channl_type.translations[param_key]["translated_name"]: param_value}
        mech_param_dict = {}
        if mechanism_name == "pas":
            channel_type = self.ion_channels["pas"]
            for param_name, its_value in other_parameters["pas"].items():
                mech_param_dict.update(get_dict_for_pas(channel_type, param_name, its_value))
        elif mechanism_name == "hh":
            for channel_name, ion_channel in self.ion_channels.items():
                if ion_channel.model == "hh":
                    channel_type = self.ion_channels[channel_name]
                    for param_name, its_value in other_parameters[channel_name].items():
                        mech_param_dict.update(
                            {channel_type.conductance_density_parameter: its_value.value})
            mech_param_dict.update({'gl': 0}) # avoids duplication with pas THIS MAY NOT BE NEEDED FOR FUTURE Arbor
        return mech_param_dict
    ######################################

    #self.setup_channel_mechanism(other_parameters) # returns self.mechanisms
    #self.attach_mechanisms(other_parameters)
    ######################################
    # Function to insert channel mechanisms
    ######################################
    def attach_mechanism(self, other_parameters):
        for mechanism_name, the_mechanism in self.mechanisms.items():
            for channel_name, ion_channel in self.ion_channels.items():
                if ion_channel.model == mechanism_name:
                    region = other_parameters[channel_name]["conductance_density"].selector
                    for chnnl_name, ion_chnnl in self.ion_channels.items():
                        param_region = other_parameters[chnnl_name]["conductance_density"].selector
                        if param_region == region:
                            if "e_rev" in other_parameters[chnnl_name].keys():
                                self.cell.paint('"{}"'.format(region), chnnl_name,
                                                rev_pot=other_parameters[chnnl_name]["e_rev"])
                    self.cell.paint('"{}"'.format(region), the_mechanism)
    ######################################

    def memb_init(self):
        for state_var in ('v',):
            initial_value = getattr(self, '{0}_init'.format(state_var))
            assert initial_value is not None
            if state_var == 'v':
                for section in self.sections.values():
                    for seg in section:
                        seg.v = initial_value
            else:
                raise NotImplementedError()
