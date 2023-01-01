# encoding: utf-8
"""
Standard base_cells for the neuron module.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.standardmodels import cells as base_cells, build_translations
from pyNN.arbor.cells import (ArborTemplate, RandomSpikeSource)
from pyNN.morphology import Morphology, NeuriteDistribution

from .simulator import state  # Hack for SpikeSourcePoisson


class MultiCompartmentNeuron(base_cells.MultiCompartmentNeuron):
    """
    **1. Inherited attributes.**

    This class is a child class of the same name inheriting the following public attributes:

    +------------------------------+-----------------------------------------------------+
    | Attribute name               | Attribute value                                     |
    +==============================+=====================================================+
    | `default_parameters`         | `{"morphology": None, "cm": uniform('all', 1.0),`   |
    |                              | `"Ra": 35.4, "ionic_species": None}`                |
    +------------------------------+-----------------------------------------------------+
    | `recordable`                 | `['spikes']`                                        |
    +------------------------------+-----------------------------------------------------+
    | `injectable`                 | `True`                                              |
    +------------------------------+-----------------------------------------------------+
    | `receptor_types`             | `()`                                                |
    +------------------------------+-----------------------------------------------------+
    | `units`                      | `{'v': 'mV', 'gsyn_exc': 'uS', 'gsyn_inh': 'uS',`   |
    |                              | `'na.m': 'dimensionless', 'na.h': 'dimensionless',` |
    |                              | `'kdr.n': 'dimensionless'}`                         |
    +------------------------------+-----------------------------------------------------+
    | `translate`                  | translator method that takes parameters as input    |
    +------------------------------+-----------------------------------------------------+

    **2. Verify necessary data conditions.**

    ::
       import pyNN.arbor as sim

       sim.setup(timestep=0.025)

       your_cell_class = sim.MultiCompartmentNeuron.setup(
                             label="MyCell",
                             ion_channels={'pas': sim.PassiveLeak,
                                           'na': sim.NaChannel,
                                           'kdr': sim.KdrChannel},
                             ionic_species={'na_ion': sim.NaIon,
                                            'k_ion': sim.KIon},
                             post_synaptic_entities={'AMPA': sim.CondExpPostSynapticResponse,
                                                     'GABA_A': sim.CondExpPostSynapticResponse})

       your_cell_class(
                    morphology=pyr_morph,
                    pas={"conductance_density": uniform('all', 0.0003),
                         "e_rev":-54.3},
                    na={"conductance_density": uniform('soma', 0.120),
                        "e_rev": 50.0},
                    kdr={"conductance_density": by_distance(apical_dendrites(), lambda d: 0.05*d/200.0),
                         "e_rev": -77.0},
                    na_ion={"reversal_potential"=50.0},
                    k_ion={"reversal_potential"=-77.0},
                    cm=1.0,
                    Ra=500.0,
                    AMPA={"density": uniform('all', 0.05),  # number per µm
                          "e_rev": 0.0,
                          "tau_syn": 2.0},
                    GABA_A={"density": by_distance(dendrites(), lambda d: 0.05 * (d < 50.0)),  # number per µm
                            "e_rev": -70.0,
                            "tau_syn": 5.0})
    """
    # translations of the possible parameters
    translations = build_translations(('Ra', 'rL'),  # (pynn_name, sim_name)
                                      ('cm', 'cm'),
                                      ('morphology', 'morphology'),
                                      # ('ionic_species', 'ionic_species'),
                                      ('na_ion', 'na'),
                                      ('k_ion', 'k'),
                                      ('ca_ion', 'ca'), )
    default_initial_values = {}

    # Initialize the attributes so that if their parameters are not passed (during instantiation)
    # error that the attribute is not available does not occur.
    ion_channels = {}
    ionic_species = {}
    post_synaptic_entities = {}

    def __init__(self, **parameters):
        # replace classes with instantiated objects
        for name, ion_channel in self.ion_channels.items():
            self.ion_channels[name] = ion_channel(**parameters.pop(name))
        for name, ionic_species in self.ionic_species.items():
            # name = name.split('_')[0] # remove the trailing _ion named as parameter
            # the ion x is named as x_ion (trailing _ion) when defining its parameter
            # self.ionic_species[name] = ionic_species(name, **parameters.pop(name+"_ion"))
            # self.ionic_species[name] = ionic_species(**parameters.pop(name + "_ion"))
            self.ionic_species[name] = ionic_species(**parameters.pop(name))
            # naming key for ionic_species and its parameters the same i.e x_ion avoids confusion
            # Also, the form is x_ion because the key x often conflicts with channel name.
        for name, pse in self.post_synaptic_entities.items():
            self.post_synaptic_entities[name] = pse(**parameters.pop(name))
        # Extend the __init__ inherited from the base MultiCompartmentNeuron
        # print(self.parameter_space) # parameter_space is not created yet
        super(MultiCompartmentNeuron, self).__init__(**parameters)  # parameter_space is created
        # print("just created parameter_space")
        # print(self.parameter_space)
        # and add parameter space of the class objects into
        # respective key of the parameter space
        for name, ion_channel in self.ion_channels.items():
            self.parameter_space[name] = ion_channel.parameter_space
        for name, ionic_species in self.ionic_species.items():
            # self.parameter_space[name+"_ion"] = ionic_species.parameter_space
            # IonicSpecies is not a model (i.e. grandchild of StandardModelType(models.BaseModelType))
            self.parameter_space[name] = ionic_species.parameter_space
        for name, pse in self.post_synaptic_entities.items():
            self.parameter_space[name] = pse.parameter_space
        # print(self.parameter_space)  # new keys added to parameter_space
        # print(self.parameter_space.schema)  # FIX ME (elsewhere): Output changes with native_parameters method
        # See ~/arbor/populations where native_parameters is called.
        # This method is defined in ~/pyNN/standardmodels/__init__.py
        # First, mycell.parameter_space.schema == mycell.get_schema()
        # Then, mycell.translate(parameter_space) changes mycell.parameter_space.schema but how?
        self.extra_parameters = {}
        self.spike_source = None
        # Below are necessary attributes of ArborTemplate needed to build the cell otherwise it is inaccessible
        self.morphology = None
        self._arbor_morphology = None
        self._arbor_tree = None
        self._arbor_labels = None
        self._decor = None

    @classmethod
    def setup(cls, **attributes):
        # returns the class
        mc_attributes = ["label", "ion_channels", "ionic_species", "post_synaptic_entities"]
        [setattr(cls, keyword, attributes[keyword])
         for keyword in mc_attributes if keyword in attributes]
        return cls

    def get_schema(self):
        schema = {
            "morphology": Morphology,
            "cm": NeuriteDistribution,
            "Ra": float,
        }
        for name, ion_channel in self.ion_channels.items():
            # print("channel name")
            # print(name)
            schema[name] = ion_channel.get_schema()
        for name, ionic_species in self.ionic_species.items():
            # print("ion name")
            # print(name)
            schema[name] = ionic_species.get_schema()
        for name, pse in self.post_synaptic_entities.items():
            schema[name] = pse.get_schema()
        # print("get_schema() output in arbors MultiCompartmentNeuron")
        # import pprint
        # pprint.pprint(schema)
        return schema

    @property  # can you have a classmethod-like property?
    def default_parameters(self):
        return {}

    @property
    def segment_names(self):  # rename to section_names?
        return [seg.name for seg in self.morphology.segments]

    # def __getattr__(self, item):
    #    if item in self.segment_names:
    #        return Segment(item, self)

    def has_parameter(self, name):
        """Does this model have a parameter with the given name?"""
        return False  # todo: implement this

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
                    (ArborTemplate,),
                    {"ion_channels": self.ion_channels,
                     "ionic_species": self.ionic_species,
                     "post_synaptic_entities": self.post_synaptic_entities,
                     "morphology": self.morphology,
                     "_arbor_morphology": self._arbor_morphology,
                     "_arbor_labels": self._arbor_labels,
                     "_decor": self._decor})


# class SpikeSourcePoisson(base_cells.SpikeSourcePoisson):
#     __doc__ = base_cells.SpikeSourcePoisson.__doc__
#
#     translations = build_translations(
#         ('start', 'start'),
#         ('rate', 'frequency',),
#         ('duration', 'duration'),
#     )
#     model = RandomSpikeSource

class SpikeSourcePoisson(object):

    translations = build_translations(
        ('start', 'start'),
        ('rate', 'frequency',),
        ('duration', 'duration'),
    )

    def __init__(self, start=0, rate=1e12, duration=0):
        # PyNN units = {'duration': 'ms', 'rate': 'Hz', 'start': 'ms'}
        self.duration = duration  # ditto NEURON backend
        self.noise = 1  # ditto NEURON backend
        self.source = self  # ditto NEURON backend
        self.tstart = start
        self.freq = rate/1000  # KHz for Arbor
        # self.seed = np.random.randint(0, 100)
        self.seed = state.mpi_rank + state.native_rng_baseseed
        self.sched = arbor.poisson_schedule(self.tstart, self.freq, self.seed)
        self.source = "spike_source"  # Should this be assigned in specific standardmodel spike source class?
