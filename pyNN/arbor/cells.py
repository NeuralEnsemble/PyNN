
from collections import defaultdict
import arbor
from neuroml import Point3DWithDiam
from ..morphology import uniform, NeuroMLMorphology, MorphIOMorphology
from ..models import BaseCellType
from morphio import SectionType


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


def build_cable_cell_parameters(parameters, ion_channels):
    std_morphology = parameters["morphology"].base_value  # todo: should evaluate the value

    # label dictionary
    labels = {
        "all": "(all)",
        "soma": "(tag 1)",
        "axon": "(tag 2)",
        "dend": "(tag 3)",
        "basal_dendrite": "(tag 4)",
        "apical_dendrite": "(tag 4)",
        "root": "(root)",
        "mid-dend": "(location 0 0.5)"
    }

    # tree
    if isinstance(std_morphology, NeuroMLMorphology):
        tree = arbor.segment_tree()
        for i, segment in enumerate(std_morphology.segments):
            prox = convert_point(segment.proximal)
            dist = convert_point(segment.distal)
            tag = region_name_to_tag(segment.name)
            if segment.parent is None:
                parent = arbor.mnpos
            else:
                parent = segment.parent.id
            tree.append(parent, prox, dist, tag=tag)
            if segment.name not in labels:
                labels[segment.name] = f"(segment {i})"
    elif isinstance(std_morphology, MorphIOMorphology):
        tree = arbor.load_swc_neuron(std_morphology.morphology_file, raw=True)
    else:
        raise ValueError("{} not supported as a neuron morphology".format(type(std_morphology)))

    mechanism_parameters = defaultdict(lambda: defaultdict(dict))
    for mechanism_name, ion_channel in ion_channels.items():
        ion_channel_parameters = ion_channel.native_parameters
        native_name = ion_channel.get_model(ion_channel_parameters)
        for pname, pval in ion_channel_parameters.items():
            if isinstance(pval.base_value, uniform):
                if isinstance(pval.base_value.selector, str):
                    region = f'"{pval.base_value.selector}"'
                    mechanism_parameters[native_name][region][pname] = pval.base_value.value
                else:
                    raise NotImplementedError()
            elif isinstance(pval.base_value, (int, float)):
                region = '"all"'
                mechanism_parameters[native_name][region][pname] = pval.base_value
            else:
                raise NotImplementedError
        # todo: handle the case where different parameters of the same mechanism have different region specs

    # decor
    def build_decor(i):
        decor = arbor.decor()
        # Set the default properties of the cell (this overrides the model defaults).
        decor.set_property(
            cm=parameters["cm"].base_value * 0.01,  # µF/cm² -> F/m²
            rL=parameters["Ra"].base_value * 1      # Ω·cm
        )
        for ion_name, ionic_species in parameters["ionic_species"].base_value.items():
            assert ion_name == ionic_species.ion_name
            decor.set_ion(ion_name,
                        int_con=ionic_species.internal_concentration,
                        ext_con=ionic_species.external_concentration,
                        rev_pot=ionic_species.reversal_potential) #method="nernst/na")
        for native_name, region_params in mechanism_parameters.items():
            for region, params in region_params.items():
                if native_name == "hh":
                    params["gl"] = 0.0
                decor.paint(region, arbor.density(native_name, params))

        policy = arbor.cv_policy_max_extent(10.0)  # to do: allow user to specify this value and/or the policy more generally
        decor.discretization(policy)

        return decor

    return {
        "tree": tree,
        "decor": build_decor,
        "labels": arbor.label_dict(labels)
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
