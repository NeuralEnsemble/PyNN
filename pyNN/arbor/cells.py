# encoding: utf-8
"""
Definition of cell classes for the neuron module.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from math import pi
from collections import defaultdict
from functools import reduce
import numpy as np
import neuroml
import arbor
import re
import copy
#import numpy.random

from pyNN import errors
# from pyNN.models import BaseCellType
from pyNN.morphology import NeuriteDistribution, IonChannelDistribution, MorphologyFilter
import pyNN.morphology as pyNNmorph
from .standardmodels.ion_channels import LeakyChannel
# from .recording import recordable_pattern
from .simulator import state

from pyNN.arbor.procedures.step1 import CreateBackendSegments
from pyNN.arbor.procedures.step2 import ConfigureMorphology
from pyNN.arbor.procedures.step3 import CreateArborMorphology
from pyNN.arbor.procedures.step4 import CreateArborLabels
from pyNN.arbor.procedures.step5 import DecorateIonicSpecies
from pyNN.arbor.procedures.step6 import DecorateIonChannels
from pyNN.arbor.procedures.step7 import DecorateSynapse

import logging

logger = logging.getLogger("PyNN")


class ArborTemplate(object):

    def __init__(self, morphology, cm, Ra, **other_parameters):
        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None
        # self.sections = {}  # Does not fix TypeError due to HasSections which is a metaclass of MultiCompartmentNeuron in pyNN/standardmodels/cells.py
        # self.spike_times = h.Vector(0)  # change this for arbor

        # set morphology
        # print(morphology)
        # print(hasattr(morphology, '_morphology'))  #HACK: Because parameter_space["morphology"].base_value.item()
        self.morphology = morphology.item()  # Since, morphology = parameter_space["morphology"].base_value
        # print(self.morphology)  # show the fix be made in native_parameters method in pyNN/parameters.py file
        # print(self.morphology._morphology)
        if isinstance(self.morphology._morphology, neuroml.arraymorph.ArrayMorphology):  # For swc loaded morphology
            # creates self.morphology.backend_segments, adjusts them and adds names
            # self.morphology = CreateBackendSegments.adjust_segments(self.morphology)
            # self.morphology = CreateBackendSegments.add_segment_names(self.morphology)
            self.morphology = CreateBackendSegments(self.morphology)
        else:
            setattr(self.morphology, "backend_segments", self.morphology.segments)
        # Update self.morphology by adding attribute self.morphology._soma_index
        self.morphology = ConfigureMorphology.set_soma_index(self.morphology)
        #
        if isinstance(self.morphology._morphology, neuroml.Morphology) \
                and not isinstance(self.morphology._morphology, neuroml.arraymorph.ArrayMorphology):
            # create self.morphology.section_groups
            self.morphology = ConfigureMorphology.include_section_groups(self.morphology)

        # create cable cell
        # creates self._arbor_morphology & self._arbor_tree
        self._arbor_tree, self._arbor_morphology = CreateArborMorphology(self.morphology)
        # creates self._arbor_labels
        self._arbor_labels = CreateArborLabels(self.morphology)
        # self.sections = {} # is this needed for arbor?
        # self.section_labels = {} # dito

        # Create decorations
        # for ky in other_parameters.keys():
        #     is_ion = re.search(".*ion$", ky)
        #     if is_ion:
        #         self._decorate_ionic_species(other_parameters)  # updates self.__decor
        self._decor = arbor.decor()
        self._decor.set_property(cm=cm, rL=Ra)
        # UPDATES self._decor
        # arg_list = DecorateIonicSpecies(self.ionic_species, other_parameters)
        # [self._decor.set_ion(tup) for tup in arg_list]  # decor.set_ion(ion_species.model, **dict_input)
        self._decor = DecorateIonicSpecies(self._decor, self.ionic_species, other_parameters)
        # UPDATES self._arbor_labels & self._decor
        self._arbor_labels, self._decor = DecorateIonChannels(self._decor, self._arbor_labels, self.ion_channels,
                                                              other_parameters)
        self._arbor_labels, self._decor = DecorateSynapse(self._decor, self._arbor_labels, self.post_synaptic_entities,
                                                          other_parameters, self._arbor_morphology)

    # ###############################
    # # Functions to decorate synapse
    # ###############################
    # def __append_synapse_labels(self):
    #     """
    #     Call this as self.__append_synapse_labels() to append synapse labels into self._arbor_labels
    #     """
    #     self._arbor_labels.append(arbor.label_dict(self.__create_synapse_definitions()))
    #
    # def __create_synapse_definitions(self, other_parameters):
    #     # Arbor uses a domains specific language (DSL) to describe regions and locations, which are given labels.
    #     dict_defs = {}
    #     for name, pse in self.post_synaptic_entities.items():
    #         region_name = self._get_pse_region_name(name, other_parameters)
    #         synapse_site = "synapse_" + region_name
    #         if re.search("soma", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         elif re.search("axon", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         elif re.search("basal", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         elif (re.search("apical", region_name, re.IGNORECASE)
    #               and re.search("dend", region_name, re.IGNORECASE)):
    #             dict_defs.update({synapse_site: ""})
    #         elif re.search("custom", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         elif re.search("neurite", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         elif re.search("glia", region_name, re.IGNORECASE):
    #             dict_defs.update({synapse_site: ""})
    #         else:
    #             dict_defs.update({synapse_site: "(on-branches 0.5)"})  # DEFAULT is mid-point of all branches
    #             # Another choice is uniformly distributed random locations, for 10 random locations
    #             # "(uniform (all) 0 9 0)" last integer (here zero) is seed number.
    #             # "(on-branches 0.5)" is the default because `from pyNN.morphology import uniform`
    #             # does not mean uniform distribution, instead "uniform" in the context of density
    #             # should be thought of as homegeneity since the synapse are homogeneously distributed
    #             # across all branches or every branch of a specific tag (say, dendrites) such that
    #             # the synapse location is given by the second argument of the density function
    #             # (here, uniform function).
    #     return dict_defs
    #
    # # AMPA={"density": uniform('all', 0.05),  # number per µm
    # #       "e_rev": 0.0,
    # #       "tau_syn": 2.0},
    # # GABA_A={"density": by_distance(dendrites(), lambda d: 0.05 * (d < 50.0)),  # number per µm
    # #         "e_rev": -70.0,
    # #         "tau_syn": 5.0}
    # def __get_pse_region_name(self, pse_name, other_parameters):
    #     pse_density_site = other_parameters[pse_name]["density"].selector
    #     if isinstance(pse_density_site, str):
    #         return pse_density_site
    #     else:
    #         return pse_density_site.__class__.__name__

    ########### END OF FUNCTION ##############

    # def memb_init(self):
    #     for state_var in ('v',):
    #         initial_value = getattr(self, '{0}_init'.format(state_var))
    #         assert initial_value is not None
    #         if state_var == 'v':
    #             for section in self.sections.values():
    #                 for seg in section:
    #                     seg.v = initial_value
    #         else:
    #             raise NotImplementedError()


class RandomSpikeSource(object):

    parameter_names = ('start', 'frequency', 'duration')

    def __init__(self, start=0, frequency=1e12, duration=0):
        # PyNN units = {'duration': 'ms', 'rate': 'Hz', 'start': 'ms'}
        self.duration = duration  # ditto NEURON backend
        self.noise = 1  # ditto NEURON backend
        self.source = self  # ditto NEURON backend
        self.tstart = start
        self.freq = frequency/1000  # KHz for Arbor
        # self.seed = np.random.randint(0, 100)
        self.seed = state.mpi_rank + state.native_rng_baseseed
        # self.sched = arbor.poisson_schedule(tstart, freq, seed)
        self.source = "spike_source"  # Should this be assigned in specific standardmodel spike source class?

    # def _set_interval(self, value):
    #     self.switch.weight[0] = -1
    #     self.switch.event(h.t + 1e-12, 0)
    #     self.interval = value
    #     self.switch.weight[0] = 1
    #     self.switch.event(h.t + 2e-12, 1)
    #
    # def _get_interval(self):
    #     return self.interval
    # _interval = property(fget=_get_interval, fset=_set_interval)
