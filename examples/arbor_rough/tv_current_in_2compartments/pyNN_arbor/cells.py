# ~/mc/pyNN/arbor/cells.py
# encoding: utf-8
"""
Definition of cell classes for the Arbor module.
:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from math import pi
from collections import defaultdict
import numpy
#from neuron import h, nrn, hclass
import arbor
import re
import numpy.random

from pyNN import errors
from pyNN.models import BaseCellType
from pyNN.morphology import NeuriteDistribution, IonChannelDistribution
#from .recording import recordable_pattern
from .simulator import state

try:
    reduce
except NameError:
    from functools import reduce

logger = logging.getLogger("PyNN")


def _new_property(obj_hierarchy, attr_name):
    """
    Returns a new property, mapping attr_name to obj_hierarchy.attr_name.
    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """

    def set(self, value):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value)

    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        return getattr(obj, attr_name)
    return property(fset=set, fget=get)


class NativeCellType(BaseCellType):

    def can_record(self, variable, location=None):
        # crude check, could be improved
        #return bool(recordable_pattern.match(variable))
        pass

    # todo: use `guess_units` to construct "units" attribute

class ArborCableTemplate(object):

    def __init__(self, morphology, cm, Ra, ionic_species, **other_parameters):
        import neuroml
        import neuroml.arraymorph

        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None
        #self.spike_times = h.Vector(0)

        # create Arbor cable cell
        self.morphology = morphology
        self.create_arbor_tree() # aka segment tree
        self.create_label_dict()
        self.cable_cell = arbor.cable_cell( self.arbor_tree, arbor.label_dict( self.label_dict ) )
        self.arbor_morphology = arbor.morphology( self.arbor_tree )
        #
        self.ionic_species = ionic_species
        self.other_parameters = other_parameters
        #
        # setup ion channels and attach to the cable cell
        self.set_ions()
        self.set_mechanisms()
        self.attach_mechanisms()
        #
        # Improve discretization precision to 2 um
        #self.cable_cell.compartments_length(2)

        # insert post-synaptic mechanisms
#        for name, pse in self.post_synaptic_entities.items():
#            parameters = other_parameters[name]
#            mechanism_name = pse.model
#            synapse_model = getattr(h, mechanism_name)
#            density_function = parameters["density"]
#            for index, id in enumerate(self.sections):
#                density = density_function.value_in(self.morphology, index)
#                if density > 0:
#                    n_synapses, remainder = divmod(density, 1)
#                    rnd = numpy.random  # todo: use the RNG from the parent Population
#                    if rnd.uniform() < remainder:
#                        n_synapses += 1
#                    section = self.sections[id]
#                    for i in range(int(n_synapses)):
#                        self.morphology.synaptic_receptors[name][id].append(synapse_model(0.5, sec=section))
        
        # ============================ Methods to create the cable cell ===============================
        def create_arbor_tree(self):
            tree = arbor.segment_tree()
            for i, nml_seg in enumerate(self.morphology.segments):
                self.append_arbor_tree(tree, nml_seg)
            self.arbor_tree = tree
            
        def append_arbor_tree(self, tree, nml_seg):
            if not nml_seg.parent:
                tree.append(arbor.mnpos,
                            arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                         nml_seg.proximal.diameter/2),
                            arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                         nml_seg.distal.diameter/2), tag=get_swc_tag(nml_seg))
            else:
                tree.append(nml_seg.parent.id,
                            arbor.mpoint(nml_seg.proximal.x, nml_seg.proximal.y, nml_seg.proximal.z,
                                         nml_seg.proximal.diameter/2),
                            arbor.mpoint(nml_seg.distal.x, nml_seg.distal.y, nml_seg.distal.z,
                                         nml_seg.distal.diameter/2), tag=get_swc_tag(nml_seg))
        def get_swc_tag(self, nml_seg):
            """
            Specification based on http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
            """
            if re.search("soma", nml_seg.name, re.IGNORECASE):
                return 1
            elif re.search("axon", nml_seg.name, re.IGNORECASE):
                return 2
            elif re.search("dend", nml_seg.name, re.IGNORECASE):
                return 3
            else:
                return 5
        
        def create_label_dict(self):
            """
            Arbor uses a domains specific language (DSL) to describe regions and locations, which are given labels.
            """
            self.label_dict = {}
            for i, nml_seg in enumerate(self.morphology.segments):
                self.label_dict.update({nml_seg.name: "(tag "+ str(get_swc_tag(nml_seg))+ ")"})
                self.label_dict.update({"everywhere": "(all)"})
        
        # ===================== Method to setting ion channels and attaching them =====================
        
        def set_ions(self):
            for name, ion_channel in self.ionic_species.items():
                if name != "pas":
                    self.cable_cell.set_ion( ion = ion_channel.ion_name,
                                             rev_pot = self.other_parameters[name]["e_rev"] )
        def set_mechanisms(self):
            hh_dict = {"gl": 0.0} # to avoid duplication by "pas"
            for name, ion_channel in self.ionic_species.items():
                if name == "pas":
                    self.leaky_chnl = \
                    arbor.mechanism( name,
                                    {ion_channel.translations["e_rev"]["translated_name"]:
                                     self.other_parameters[name]["e_rev"],
                                     ion_channel.translations["conductance_density"]["translated_name"]:
                                     self.other_parameters[name]["conductance_density"].value} )
                else:
                    hh_dict.append( {ion_channel.translations["conductance_density"]["translated_name"]:
                                     self.other_parameters[name]["conductance_density"].value} )
            self.hh_chnl = arbor.mechanism("hh", hh_dict)
            
        def attach_mechanisms(self):
            """
            Rationale for extra quotes: https://github.com/arbor-sim/arbor/discussions/1204#discussioncomment-117998
            """
            all_regions = (lambda s: [s+"\""+seg.name+"\") " for seg in self.morphology.segments])(" (region ")
            get_chnl = lambda chnl_name: self.leaky_chnl if chnl_name=="pas" else self.hh_chnl
            if "channel_regions" not in self.other_parameters:
                for name, ion_channel in self.ionic_species.items():
                    if name == "pas":
                        self.cable_cell.paint('"everywhere"', self.leaky_chnl)
                    else:
                        self.cable_cell.paint( "(join"+" ".join( (all_regions) )+")",
                                               ion_channel.ion_name,
                                               rev_pot = self.other_parameters[name]["e_rev"] )
            else:
                # self.other_parameters["channel_regions"] = {"soma": ["na", "kdr", "pas"], "dendrite": ["na"]}
                for region_name, chnl_name_list in self.channel_regions.items():
                    [self.cable_cell.paint( f'"{region_name}"', self.ionic_species[channel_name].ion_name,
                                            rev_pot = self.other_parameters[channel_name]["e_rev"] )
                     for channel_name in chnl_name_list  if channel_name != "pas"]
                    if "na" and "kdr" in chnl_name_list:
                        chnl_name_list.remove("kdr")
                    [self.cable_cell.paint( f'"{region_name}"', get_chnl(channel_name) )
                     for channel_name in chnl_name_list]
        
        # Set stimulus
        #cell.place('"stim"', arbor.iclamp(10, 2, 0.8))
        #cell.place('"stim"', arbor.iclamp(50, 2, 0.8))
        #cell.place('"stim"', arbor.iclamp(80, 2, 0.8))
        
        # ===================== Method to set a spike source from the cable cell ======================
        
        def set_spike_detector(self):
            if self.spike_source:
                if "spike_threshold" not in self.other_parameters:
                    self.cable_cell(f'"{self.spike_source}"', arbor.spike_detector(-10))
                else:
                    self.cable_cell(f'"{self.spike_source}"',
                                    arbor.spike_detector( self.other_parameters["spike_threshold"][self.spike_source] ))
            elif "axon_initial_segment" in self.label_dict:
                if "spike_threshold" not in self.other_parameters:
                    self.cable_cell('"axon_initial_segment"', arbor.spike_detector(-10))
                else:
                    self.cable_cell('"axon_initial_segment"',
                                    arbor.spike_detector( self.other_parameters["spike_threshold"]["axon_initial_segment"] ))
            else: # if self.spike_source is None (DEFAULT)
                if "spike_threshold" not in self.other_parameters:
                    self.cable_cell("(location 0 0.5)", arbor.spike_detector(-10))
                else:
                    self.cable_cell("(location 0 0.5)", arbor.spike_detector( self.other_parameters["spike_threshold"]["root"] ))
            #self.source Arbor does NOT need self.source and self.rec

#    def memb_init(self):
#        for state_var in ('v',):
#            initial_value = getattr(self, '{0}_init'.format(state_var))
#            assert initial_value is not None
#            if state_var == 'v':
#                for section in self.sections.values():
#                    for seg in section:
#                        seg.v = initial_value
#            else:
#                raise NotImplementedError()