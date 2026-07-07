
from collections import defaultdict
from lazyarray import larray
import arbor
from arbor import units as U
from neuroml import Point3DWithDiam
from ..morphology import Morphology, NeuroMLMorphology, MorphIOMorphology, IonChannelDistribution
from ..models import BaseCellType
from ..parameters import ParameterSpace


# Units for the parameters of current-source mechanisms (e.g. iclamp).
ELECTRODE_PARAM_UNITS = {
    "tstart": U.ms,
    "duration": U.ms,
    "current": U.nA,
}


def convert_point(p3d: Point3DWithDiam) -> arbor.mpoint:
    return arbor.mpoint(p3d.x, p3d.y, p3d.z, p3d.diameter/2)


def region_name_to_tag(name):
    map = {
        "soma": 1,
        "axon": 2,
        "dendrite": 3,
        "apical dendrite": 4
    }
    # "arbitrary tags, including zero and negative values, can be used"
    return map.get(name, -1)


class CellDescriptionBuilder:

    def __init__(self, parameters, ion_channels, post_synaptic_entities=None):
        assert isinstance(parameters, ParameterSpace)
        self.parameters = parameters
        self.ion_channels = ion_channels
        self.post_synaptic_entities = post_synaptic_entities
        self.labels = {
            "all": "(all)",
            "soma": "(tag 1)",
            "axon": "(tag 2)",
            "dend": "(tag 3)",
            "basal_dendrite": "(tag 4)",
            "apical_dendrite": "(tag 4)",
            "root": "(root)",
            "mid-dend": "(location 0 0.5)"
        }
        self.initial_values = {}
        self.current_sources = defaultdict(list)

    def _build_tree(self, i):
        self.parameters["morphology"].dtype = Morphology
        std_morphology = self.parameters["morphology"]._partially_evaluate(i, simplify=True)  # evaluates the larray
        if isinstance(std_morphology, NeuroMLMorphology):
            tree = arbor.segment_tree()
            for i, segment in enumerate(std_morphology.segments):
                prox = convert_point(segment.proximal)
                dist = convert_point(segment.distal)
                tag = region_name_to_tag(segment.name)
                if tag == -1 and std_morphology.section_groups:
                    for section_type, id_list in std_morphology.section_groups.items():
                        if i in id_list:
                            tag = section_type.value
                if segment.parent is None:
                    parent = arbor.mnpos
                else:
                    parent = segment.parent.id
                tree.append(parent, prox, dist, tag=tag)
                if segment.name not in self.labels:
                    self.labels[segment.name] = f"(segment {i})"
        elif isinstance(std_morphology, MorphIOMorphology):
            tree = arbor.load_swc_neuron(std_morphology.morphology_file).segment_tree
        else:
            raise ValueError("{} not supported as a neuron morphology".format(type(std_morphology)))

        return tree

    def _build_decor(self, i):
        mechanism_parameters = defaultdict(lambda: defaultdict(dict))
        for mechanism_name, ion_channel in self.ion_channels.items():
            ion_channel_parameters = ion_channel.native_parameters.evaluate(mask=[i], simplify=True)
            native_name = ion_channel.get_model(ion_channel_parameters)
            for pname, pval in ion_channel_parameters.items():
                if isinstance(pval, IonChannelDistribution):
                    region, value = pval.resolve()
                    mechanism_parameters[native_name][region][pname] = value
                elif isinstance(pval, (int, float)):
                    region = '"all"'
                    mechanism_parameters[native_name][region][pname] = pval
                else:
                    raise NotImplementedError
            # todo: handle the case where different parameters of the same mechanism have different region specs

        decor = arbor.decor()
        # Set the default properties of the cell (this overrides the model defaults).
        decor.set_property(
            cm=self.parameters["cm"][i] * U.uF / U.cm2,
            rL=self.parameters["Ra"][i] * U.Ohm * U.cm,
            Vm=self.initial_values["v"][i] * U.mV
        )
        if not self.parameters["ionic_species"]._evaluated:
            self.parameters["ionic_species"].evaluate(simplify=True)
        for ion_name, ionic_species in self.parameters["ionic_species"].items():
            assert ion_name == ionic_species.ion_name
            ion_kwargs = {}
            if ionic_species.internal_concentration is not None:
                ion_kwargs["int_con"] = ionic_species.internal_concentration * U.mM
            if ionic_species.external_concentration is not None:
                ion_kwargs["ext_con"] = ionic_species.external_concentration * U.mM
            if ionic_species.reversal_potential is not None:
                ion_kwargs["rev_pot"] = ionic_species.reversal_potential * U.mV
            decor.set_ion(ion_name, **ion_kwargs)  # method="nernst/na")
        for native_name, region_params in mechanism_parameters.items():
            for region, params in region_params.items():
                if native_name == "hh":
                    params["gl"] = 0.0
                decor.paint(region, arbor.density(native_name, params))
        # insert post-synaptic mechanisms
        morph = self.parameters["morphology"]._partially_evaluate([i], simplify=True)
        if self.post_synaptic_entities:
            for name, pse in self.post_synaptic_entities.items():
                pse_parameters = pse.native_parameters.evaluate([i], simplify=True)
                location_generator = pse_parameters.pop("locations")
                # todo: handle setting other synaptic parameters
                locations = location_generator.generate_locations(morph, label=name)
                assert isinstance(locations, list)
                for (locset, label) in locations:
                    self.labels[label] = locset
                    decor.place(locset, arbor.synapse(pse.model, pse_parameters.as_dict()), label)

        # insert current sources
        for current_source in self.current_sources[i]:
            location_generator = current_source["location_generator"]
            mechanism = getattr(arbor, current_source["model_name"])
            for locset, label in location_generator.generate_locations(morph, label=f"{current_source['model_name']}_label"):
                params = current_source["parameters"].evaluate(simplify=True).as_dict()
                params = {
                    name: value * ELECTRODE_PARAM_UNITS[name] if name in ELECTRODE_PARAM_UNITS else value
                    for name, value in params.items()
                }
                mech = mechanism(**params)
                decor.place(locset, mech, label)

        # add spike source
        decor.place('"root"', arbor.threshold_detector(-10 * U.mV), "detector")
        # todo: allow user to choose location and threshold value

        policy = arbor.cv_policy_max_extent(10.0)
        # to do: allow user to specify this value and/or the policy more generally
        decor.discretization(policy)

        return decor

    def set_initial_values(self, variable, initial_values):
        assert isinstance(initial_values, larray)
        self.initial_values[variable] = initial_values

    def add_current_source(self, model_name, location_generator, index, parameters):
        for i in index:
            self.current_sources[i].append({
                "model_name": model_name,
                "parameters": parameters,
                "location_generator": location_generator
            })

    def set_shape(self, value):
        self.parameters.shape = value
        for ion_channel in self.ion_channels.values():
            ion_channel.parameter_space.shape = value
        for pse in self.post_synaptic_entities.values():
            pse.parameter_space.shape = value

    def __call__(self, i):
        return {
            "tree": self._build_tree(i),
            "decor": self._build_decor(i),
            "labels": arbor.label_dict(self.labels)
        }


class NativeCellType(BaseCellType):
    arbor_cell_kind = arbor.cell_kind.cable
    units = {"v": "mV"}

    def __init__(self, **parameters):
        self.parameter_space = parameters

    def can_record(self, variable, location=None):
        return True  # todo: implement this properly

    def describe(self, template=None):
        return "Native Arbor model"
