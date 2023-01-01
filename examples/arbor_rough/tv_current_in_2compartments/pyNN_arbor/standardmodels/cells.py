# ~/mc/pyNN/arbor/standardmodels/cells.py
# encoding: utf-8
"""
Standard base_cells for the Arbor module.
:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
from pyNN.models import BaseCellType
from pyNN.parameters import ParameterSpace
from pyNN.standardmodels import cells as base_cells, build_translations
#from pyNN.neuron.cells import (StandardIF, SingleCompartmentTraub,
#                               RandomSpikeSource, VectorSpikeSource,
#                               RandomGammaSpikeSource,
#                               RandomPoissonRefractorySpikeSource,
#                               BretteGerstnerIF, GsfaGrrIF, Izhikevich_,
#                               GIFNeuron, NeuronTemplate)
from pyNN_arbor.cells import (ArborTemplate)
from pyNN.morphology import Morphology, NeuriteDistribution
import logging


logger = logging.getLogger("PyNN")

class MultiCompartmentNeuron(base_cells.MultiCompartmentNeuron):
    """
    """
    translations = build_translations(
        ('Ra', 'rL'),  # (pynn_name, sim_name)
        ('cm', 'cm'),
        ('morphology', 'morphology'),
        ('ionic_species', 'ionic_species')
    )
    default_initial_values = {}
    ion_channels = {}
    post_synaptic_entities = {}

    def __init__(self, **parameters):
        pass
        # replace ion channel classes with instantiated ion channel objects
        for name, ion_channel in self.ion_channels.items():
            self.ion_channels[name] = ion_channel(**parameters.pop(name))
        # ditto for post synaptic responses
#        for name, pse in self.post_synaptic_entities.items():
#            self.post_synaptic_entities[name] = pse(**parameters.pop(name))
        super(MultiCompartmentNeuron, self).__init__(**parameters)
        for name, ion_channel in self.ion_channels.items():
            self.parameter_space[name] = ion_channel.parameter_space
#        for name, pse in self.post_synaptic_entities.items():
#            self.parameter_space[name] = pse.parameter_space

        self.extra_parameters = {}
        self.spike_source = None

    def get_schema(self):
        schema = {
            "morphology": Morphology,
            "cm": NeuriteDistribution,
            "Ra": float,
            "ionic_species": dict
        }
        #for name, ion_channel in self.ion_channels.items():
        #    schema[name] = ion_channel.get_schema()
        return schema

    @property   # can you have a classmethod-like property?
    def default_parameters(self):
        return {}

    @property
    def segment_names(self):  # rename to section_names?
        return [seg.name for seg in self.morphology.segments]

    #def __getattr__(self, item):
    #    if item in self.segment_names:
    #        return Segment(item, self)

    def has_parameter(self, name):
        """Does this model have a parameter with the given name?"""
        return False   # todo: implement this

    def get_parameter_names(self):
        """Return the names of the parameters of this model."""
        raise NotImplementedError

    @property
    def recordable(self):
        raise NotImplementedError

    def can_record(self, variable, location=None):
        return True  # todo: implement this properly

    @property
    def receptor_types(self):
        return self.post_synaptic_entities.keys()

    @property
    def model(self):
        return type(self.label,
                    (ArborCableTemplate,),
                    {"ion_channels": self.ion_channels,
                     "post_synaptic_entities": self.post_synaptic_entities})

    # @classmethod
    # def insert(cls, sections=None, **ion_channels):
    #     for name, mechanism in ion_channels.items():
    #         if name in cls.ion_channels:
    #             assert cls.ion_channels[name]["mechanism"] == mechanism
    #             cls.ion_channels[name]["sections"].extend(sections)
    #         else:
    #             cls.ion_channels[name] = {
    #                 "mechanism": mechanism,
    #                 "sections": sections
    #             }