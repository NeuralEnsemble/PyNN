"""
Standard cells for the Arbor module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from copy import deepcopy

import arbor

from ..standardmodels import cells, ion_channels, synapses, electrodes, build_translations
from ..parameters import ParameterSpace
from ..morphology import Morphology, NeuriteDistribution, uniform
from .cells import build_cable_cell_parameters
from .simulator import state

logger = logging.getLogger("PyNN")


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'START'),
        ('rate',     'INTERVAL',  "1000.0/rate",  "1000.0/INTERVAL"),
        ('duration', 'DURATION'),
    )
    arbor_cell_kind = arbor.cell_kind.spike_source


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'SPIKE_TIMES'),
    )
    arbor_cell_kind = arbor.cell_kind.spike_source


class BaseCurrentSource(object):
    pass


class DCSource(BaseCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )

    def inject_into(self, cells, location=None):
        if location == "soma":
            # todo: proper location mapping
            locset = '"root"'
        elif location == "dendrite":
            locset = '"mid-dend"'
        else:
            # can we be sure location is a label?
            locset = f'(on-components 0.5 (region "{location}"))'
        for cell in cells:
            start = self.native_parameters["start"].base_value
            stop = self.native_parameters["stop"].base_value
            amplitude = self.native_parameters["amplitude"].base_value
            cell.decor.place(locset, arbor.iclamp(start, stop - start, current=amplitude), "iclamp_label")


class StepCurrentSource(BaseCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )


class ACSource(BaseCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop'),
        ('frequency',  'frequency'),
        ('offset',     'offset'),
        ('phase',      'phase')
    )


class NoisyCurrentSource(BaseCurrentSource, electrodes.NoisyCurrentSource):

    translations = build_translations(
        ('mean',  'mean'),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'stdev'),
        ('dt',    'dt')
    )


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


class NaChannel(ion_channels.NaChannel):
    translations = build_translations(
        ('conductance_density', 'gnabar'),
        ('e_rev', 'ena'),
    )
    variable_translations = {
        'm': ('hh', 'm'),
        'h': ('hh', 'h')
    }
    model = "hh"
    conductance_density_parameter = 'gnabar'

    def get_model(self, parameters=None):
        return "hh"


class KdrChannel(ion_channels.KdrChannel):
    translations = build_translations(
        ('conductance_density', 'gkbar'),
        ('e_rev', 'ek'),
    )
    variable_translations = {
        'n': ('hh', 'n')
    }
    conductance_density_parameter = 'gkbar'

    def get_model(self, parameters=None):
        return "hh"


class PassiveLeak(ion_channels.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'g'),
        ('e_rev', 'e'),
    )
    conductance_density_parameter = 'g'
    global_parameters = ['e']

    def get_model(self, parameters=None):
        if parameters:
            param_entries = []
            for name, value in parameters.items():
                if name in self.global_parameters:
                    param_entries.append(f"{name}={value.base_value}")  # to fix: should be evaluated
            param_str = ",".join(param_entries)
            for name in self.global_parameters:
                parameters.pop(name, None)
            return f"pas/{param_str}"
        else:
            return "pas"


class MultiCompartmentNeuron(cells.MultiCompartmentNeuron):
    """

    """
    default_initial_values = {}
    ion_channels = {}
    post_synaptic_entities = {}
    arbor_cell_kind = arbor.cell_kind.cable

    def __init__(self, **parameters):
        # replace ion channel classes with instantiated ion channel objects
        for name, ion_channel in self.ion_channels.items():
            self.ion_channels[name] = ion_channel(**parameters.pop(name, {}))
        # ditto for post synaptic responses
        for name, pse in self.post_synaptic_entities.items():
            self.post_synaptic_entities[name] = pse(**parameters.pop(name, {}))
        super(MultiCompartmentNeuron, self).__init__(**parameters)
        for name, ion_channel in self.ion_channels.items():
            self.parameter_space[name] = ion_channel.parameter_space
        for name, pse in self.post_synaptic_entities.items():
            self.parameter_space[name] = pse.parameter_space
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

    def translate(self, parameters, copy=True):
        """Translate standardized model parameters to simulator-specific parameters."""
        if copy:
            _parameters = deepcopy(parameters)
        else:
            _parameters = parameters
        cls = self.__class__
        if parameters.schema != self.get_schema():
            # should replace this with a PyNN-specific exception type
            raise Exception(f"Schemas do not match: {parameters.schema} != {self.get_schema()}")
        # translate ion channel
        native_parameters = build_cable_cell_parameters(parameters, self.ion_channels)
        #return ParameterSpace(native_parameters, schema=None, shape=parameters.shape)
        return native_parameters

    def reverse_translate(self, native_parameters):
        raise NotImplementedError

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
                    (NeuronTemplate,),
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
