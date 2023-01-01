# encoding: utf-8
"""
Definition of cell classes for the neuron module.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from collections import defaultdict

import arbor

from pyNN.arborproto.procedures.label import base_label
from pyNN.arborproto.procedures.decorate_ionic_species import DecorateIonicSpecies
from pyNN.arborproto.procedures.decorate_ion_channels import DecorateIonChannels
from pyNN.arborproto.procedures.decorate_synapse import DecorateSynapse
from pyNN.arborproto.simulator import state

import logging

logger = logging.getLogger("PyNN")


class ArborTemplate(object):

    def __init__(self, morphology, cm, Ra, **other_parameters):
        self.traces = defaultdict(list)
        self.recording_time = False
        self.spike_source = None

        self.morphology = morphology.item()  # Since, morphology = parameter_space["morphology"].base_value

        self._arbor_morphology = arbor.load_swc_arbor(self.morphology.file)

        self._arbor_labels = base_label(self._arbor_morphology)

        # self._decor = arbor.decor()
        # self._decor.set_property(cm=cm, rL=Ra)
        # # UPDATE self._decor
        # self._decor = DecorateIonicSpecies(self._decor, self.ionic_species, other_parameters)
        # # UPDATES self._arbor_labels & self._decor
        # self._arbor_labels, self._decor = DecorateIonChannels(self._decor, self._arbor_labels, self.ion_channels,
        #                                                       other_parameters)
        # self._arbor_labels, self._decor = DecorateSynapse(self._decor, self._arbor_labels, self.post_synaptic_entities,
        #                                                   other_parameters, self._arbor_morphology)

        self._decor = arbor.decor()
        # Set the default properties of the cell (this overrides the model defaults).
        self._decor.set_property(Vm=-55)
        self._decor.set_ion("na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na")
        self._decor.set_ion("k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        # Override the cell defaults.
        self._decor.paint('"soma"', tempK=270)
        self._decor.paint('"soma"', Vm=-50)
        # Paint density mechanisms.
        self._decor.paint('"everywhere"', arbor.density("pas"))
        self._decor.paint('"soma"', arbor.density("hh"))
        self._decor.paint('"basal_dendrite"', arbor.density("hh", {"gkbar": 0.001}))
        # Synapse
        self._arbor_labels["synapse_site"] = "(location 1 0.5)"
        self._decor.place('"synapse_site"', arbor.synapse("expsyn"), "syn")
        # Attach a detector with threshold of -10 mV.
        self._decor.place('"root"', arbor.spike_detector(-10), "detector")


class RandomSpikeSource(object):
    parameter_names = ('tstart', 'frequency', 'duration')

    def __init__(self, tstart=0.0, frequency=1e12, duration=0.0):
        # PyNN units = {'duration': 'ms', 'rate': 'Hz', 'start': 'ms'}
        self.duration = duration  # ditto NEURON backend
        self.noise = 1  # ditto NEURON backend
        self.source = self  # ditto NEURON backend
        # print(tstart) don't know why but this is the parameter space
        self.tstart = tstart["start"]  # tstart
        self.freq = tstart["rate"] / 1000  # frequency / 1000  # KHz for Arbor
        # self.seed = np.random.randint(0, 100)
        self.seed = state.mpi_rank + state.native_rng_baseseed
        self.sched = arbor.poisson_schedule(self.tstart, self.freq, self.seed)
        self.source = "spike_source"  # Should this be assigned in specific standardmodel spike source class?
