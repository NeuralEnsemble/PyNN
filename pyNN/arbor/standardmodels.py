"""
Standard cells for the Arbor module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from copy import deepcopy

import arbor

from ..standardmodels import cells, ion_channels, synapses, electrodes, receptors, build_translations
from ..parameters import ParameterSpace, IonicSpecies
from ..morphology import Morphology, NeuriteDistribution, LocationGenerator
from .cells import CellDescriptionBuilder
from .simulator import state
from .morphology import LabelledLocations

logger = logging.getLogger("PyNN")


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'tstart'),
        ('rate',     'freq'),
        ('duration', 'tstop', "start + duration", "tstop - tstart"),
    )
    # todo: manage "seed"
    arbor_cell_kind = arbor.cell_kind.spike_source
    arbor_schedule = arbor.poisson_schedule


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'times'),
    )
    arbor_cell_kind = arbor.cell_kind.spike_source
    arbor_schedule = arbor.explicit_schedule


class BaseCurrentSource(object):
    pass


class DCSource(BaseCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'current'),
        ('start',      'tstart'),
        ('stop',       'duration', "stop - start", "tstart + duration")
    )

    def inject_into(self, cells, location=None):  # rename to `locations` ?
        if hasattr(cells, "parent"):
            cell_descr = cells.parent._arbor_cell_description.base_value
            index = cells.parent.id_to_index(cells.all_cells.astype(int))
        else:
            cell_descr = cells._arbor_cell_description.base_value
            index = cells.id_to_index(cells.all_cells.astype(int))
        self.parameter_space.shape = (1,)
        if location is None:
            raise NotImplementedError
        elif isinstance(location, str):
            location = LabelledLocations(location)
        elif isinstance(location, LocationGenerator):
            # morphology = cells._arbor_cell_description.base_value.parameters["morphology"].base_value  # todo: evaluate lazyarray
            # locations = location.generate_locations(morphology, label="dc_current_source")
            # assert len(locations) == 1
            # locset = locations[0]
            pass
        else:
            raise TypeError("location must be a string or a LocationGenerator")
        cell_descr.add_current_source(
            model_name="iclamp",
            location_generator=location,
            index=index,
            parameters=self.native_parameters
        )


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
        ('weight', 'weight'),
        ('delay', 'delay'),
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
        'm': ('na', 'm'),
        'h': ('na', 'h')
    }
    model = "na"
    conductance_density_parameter = 'gnabar'

    def get_model(self, parameters=None):
        return "na"


class KdrChannel(ion_channels.KdrChannel):
    translations = build_translations(
        ('conductance_density', 'gkbar'),
        ('e_rev', 'ek'),
    )
    variable_translations = {
        'n': ('kdr', 'n')
    }
    conductance_density_parameter = 'gkbar'

    def get_model(self, parameters=None):
        return "kdr"


class PassiveLeak(ion_channels.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'g'),
        ('e_rev', 'e'),
    )
    conductance_density_parameter = 'g'
    global_parameters = ['e']

    def get_model(self, parameters=None):
        if parameters:
            assert parameters._evaluated
            param_entries = []
            for name, value in parameters.items():
                if name in self.global_parameters:
                    param_entries.append(f"{name}={value}")
            param_str = ",".join(param_entries)
            for name in self.global_parameters:
                parameters.pop(name, None)
            return f"pas/{param_str}"
        else:
            return "pas"


class PassiveLeakHH(ion_channels.PassiveLeak):
    translations = build_translations(
        ('conductance_density', 'gl'),
        ('e_rev', 'el'),
    )
    conductance_density_parameter = 'gl'
    global_parameters = ['el']

    def get_model(self, parameters=None):
        if parameters:
            assert parameters._evaluated
            param_entries = []
            for name, value in parameters.items():
                if name in self.global_parameters:
                    param_entries.append(f"{name}={value}")
            param_str = ",".join(param_entries)
            for name in self.global_parameters:
                parameters.pop(name, None)
            return f"leak/{param_str}"
        else:
            return "leak"


class MultiCompartmentNeuron(cells.MultiCompartmentNeuron):
    """

    """
    default_initial_values = {}
    ion_channels = {}
    post_synaptic_entities = {}
    arbor_cell_kind = arbor.cell_kind.cable
    variable_map = {"v": "Vm"}

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
            "ionic_species": {
                "na": IonicSpecies,
                "k": IonicSpecies,
                "ca": IonicSpecies,
                "cl": IonicSpecies
            }
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
        arbor_description = CellDescriptionBuilder(_parameters, self.ion_channels, self.post_synaptic_entities)
        native_parameters = {"cell_description": arbor_description}
        return ParameterSpace(native_parameters, schema=None, shape=_parameters.shape)

    def reverse_translate(self, native_parameters):
        raise NotImplementedError

    @property   # can you have a classmethod-like property?
    def default_parameters(self):
        return {}

    @property
    def segment_names(self):  # rename to section_names?
        return [seg.name for seg in self.morphology.segments]

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


class CondExpPostSynapticResponse(receptors.CondExpPostSynapticResponse):

    translations = build_translations(
        ('locations', 'locations'),
        ('e_syn', 'e'),
        ('tau_syn', 'tau')
    )
    model = "expsyn"
    recordable = ["gsyn"]
    variable_map = {"gsyn": "g"}
