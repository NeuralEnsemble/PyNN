
from collections import defaultdict
import numpy as np
from lazyarray import larray
import arbor
from arbor import units as U
from neuroml import Point3DWithDiam
from . import _compat
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


class BaseCellDescriptionBuilder:
    """Shared protocol for the Arbor cable-cell builders consumed by
    :mod:`populations.py`.

    A builder is a callable ``builder(i)`` returning the ``tree``/``decor``/
    ``labels``/``discretization`` needed to construct the ``i``-th cell, plus
    ``set_shape``, ``set_initial_values`` and ``add_current_source``. This base
    holds only the parts common to :class:`CellDescriptionBuilder` (multicompartment)
    and :class:`PointCellDescriptionBuilder` (point neuron); ``set_shape``,
    ``_build_tree``, ``_build_decor`` and ``__call__`` are builder-specific.
    """

    def __init__(self):
        self.initial_values = {}
        self.current_sources = defaultdict(list)

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


class CellDescriptionBuilder(BaseCellDescriptionBuilder):

    def __init__(self, parameters, ion_channels, post_synaptic_entities=None):
        super().__init__()
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
            mechanism = _compat.get_electrode_mechanism(current_source["model_name"])
            for locset, label in location_generator.generate_locations(morph, label=f"{current_source['model_name']}_label"):
                params = current_source["parameters"].evaluate(simplify=True).as_dict()
                params = {
                    name: value * ELECTRODE_PARAM_UNITS[name] if name in ELECTRODE_PARAM_UNITS else value
                    for name, value in params.items()
                }
                mech = mechanism(**params)
                _compat.place_current_source(decor, locset, mech, label)

        # add spike source
        decor.place('"root"', arbor.threshold_detector(-10 * U.mV), "detector")
        # todo: allow user to choose location and threshold value

        return decor

    def set_shape(self, value):
        self.parameters.shape = value
        for ion_channel in self.ion_channels.values():
            ion_channel.parameter_space.shape = value
        for pse in self.post_synaptic_entities.values():
            pse.parameter_space.shape = value

    def __call__(self, i):
        # The discretisation cv_policy is applied by _compat.make_cable_cell at
        # cell-construction time (on the decor in Arbor 0.10, or as a cable_cell
        # argument in 0.11+). to do: allow the user to specify this value/policy.
        return {
            "tree": self._build_tree(i),
            "decor": self._build_decor(i),
            "labels": arbor.label_dict(self.labels),
            "discretization": _compat.max_extent_policy(10.0)
        }


# Geometry of the synthetic single compartment used for point neurons. The
# absolute surface area cancels out of the membrane dynamics (it is divided back
# out when deriving the specific cm/leak below), so it doesn't affect results; we
# match the pyNN.neuron backend's SingleCompartmentNeuron (L = 100 um,
# diam = 1000/pi um) so the two backends use an identical synthetic cell. Both
# NEURON and Arbor take the cylinder area to be the lateral surface only (no end
# caps), giving area = pi*L*diam = exactly 1e-3 cm2 -- a round value chosen so the
# derived specific capacitance equals the whole-cell cm numerically (1 nF -> 1
# uF/cm2), avoiding an irrational specific cm.
POINT_CELL_LENGTH_UM = 100.0
POINT_CELL_DIAMETER_UM = 1000.0 / np.pi
_POINT_CELL_AREA_CM2 = np.pi * POINT_CELL_DIAMETER_UM * POINT_CELL_LENGTH_UM * 1e-8
# A clamp conductance large enough that the post-spike reset settles within one
# timestep (tau_clamp = C_m / LIF_RESET_CONDUCTANCE << dt for any realistic C_m).
LIF_RESET_CONDUCTANCE = 1000.0  # uS


class PointCellDescriptionBuilder(BaseCellDescriptionBuilder):
    """Builds an Arbor cable cell that behaves as a PyNN point neuron.

    A point neuron (LIF, and its IF_cond_exp/IF_curr_exp specialisations) is
    realised as a single-compartment cable cell:

    * the sub-threshold leak is a ``pas`` density mechanism, with the specific
      capacitance and leak conductance derived from the whole-cell ``cm`` [nF] and
      ``tau_m`` [ms] so that the *absolute* C_m and g_leak match PyNN's values;
    * the network spike is emitted by the cell's ``threshold_detector`` (at
      ``v_thresh``);
    * the post-spike reset and refractory clamp are provided by the ``lif``
      point mechanism (see nmodl/lif.mod), driven by the detector via POST_EVENT;
    * one synapse per receptor (``expsyn`` for conductance-based,
      ``expsyn_curr`` for current-based) is placed at the soma;
    * ``i_offset`` and injected current sources become ``iclamp`` stimuli.

    Its ``__call__(i)`` / ``set_shape`` / ``set_initial_values`` /
    ``add_current_source`` interface mirrors :class:`CellDescriptionBuilder`, so a
    point-neuron Population reuses the backend's existing cable-cell machinery.

    ``neuron_parameters`` is a native :class:`ParameterSpace` with the LIF keys
    (E_L, E_R, V_th, t_ref, tau_m, C_m, i_offset). ``post_synaptic_receptors`` maps
    each receptor label to an ``(arbor_synapse_model, native_synapse_parameters)``
    pair. This is the form produced both by the composable
    :class:`~pyNN.arbor.standardmodels.PointNeuron` (from its components) and by the
    flat IF_cond_exp/IF_curr_exp standard models (from their translations).
    """

    def __init__(self, neuron_parameters, post_synaptic_receptors):
        super().__init__()
        self.neuron_parameters = neuron_parameters
        self.post_synaptic_receptors = post_synaptic_receptors
        self.shape = None

    def set_shape(self, value):
        self.shape = value
        self.neuron_parameters.shape = value
        for (_model, synapse_parameters) in self.post_synaptic_receptors.values():
            synapse_parameters.shape = value

    def _specific_properties(self, cm_nF, tau_m_ms):
        """Specific membrane capacitance [uF/cm2] and leak conductance [S/cm2]
        reproducing the whole-cell C_m [nF] and g_leak = C_m/tau_m [uS]."""
        c_spec = cm_nF * 1e-3 / _POINT_CELL_AREA_CM2
        g_spec = (cm_nF / tau_m_ms) * 1e-6 / _POINT_CELL_AREA_CM2
        return c_spec, g_spec

    def _build_tree(self, i):
        d = POINT_CELL_DIAMETER_UM
        tree = arbor.segment_tree()
        tree.append(arbor.mnpos, arbor.mpoint(0, 0, 0, d / 2),
                    arbor.mpoint(POINT_CELL_LENGTH_UM, 0, 0, d / 2), tag=1)
        return tree

    def _build_decor(self, i):
        p = self.neuron_parameters

        E_L = p["E_L"][i]
        c_spec, g_spec = self._specific_properties(p["C_m"][i], p["tau_m"][i])
        labels = {
            "all": "(all)", "soma": "(tag 1)", "root": "(root)",
        }
        decor = arbor.decor()
        decor.set_property(
            Vm=self.initial_values["v"][i] * U.mV,
            cm=c_spec * U.uF / U.cm2,
        )
        # sub-threshold leak (e is a GLOBAL parameter of pas, set via the mechanism name)
        decor.paint("(all)", arbor.density(f"pas/e={E_L}", {"g": g_spec}))
        # integrate-and-fire reset + refractory clamp
        decor.place(
            "(root)",
            arbor.synapse("lif", {"v_reset": p["E_R"][i], "t_ref": p["t_ref"][i],
                                  "g_reset": LIF_RESET_CONDUCTANCE}),
            "lif_reset",
        )
        # network spike source
        decor.place("(root)", arbor.threshold_detector(p["V_th"][i] * U.mV), "detector")

        # one synapse per receptor, labelled by receptor name so that
        # Projection.arbor_connections can match it by receptor_type prefix
        for label, (model, synapse_parameters) in self.post_synaptic_receptors.items():
            params = {key: synapse_parameters[key][i]
                      for key in synapse_parameters.keys()
                      if key != "locations"}  # a point neuron has a single location
            decor.place("(root)", arbor.synapse(model, params), label)
            labels[label] = "(root)"

        # constant offset current
        i_offset = p["i_offset"][i] if "i_offset" in p.keys() else 0.0
        if i_offset != 0.0:
            self._place_iclamp(decor, "(root)",
                               tstart=0.0, duration=1e12, current=i_offset,
                               label="i_offset")
        # injected current sources (e.g. DCSource)
        for source in self.current_sources[i]:
            location_generator = source["location_generator"]
            params = source["parameters"].evaluate(simplify=True).as_dict()
            for locset, label in location_generator.generate_locations(None, label=source["model_name"]):
                self._place_iclamp(
                    decor, locset,
                    tstart=params["tstart"], duration=params["duration"],
                    current=params["current"], label=label)

        return decor, arbor.label_dict(labels)

    def _place_iclamp(self, decor, locset, tstart, duration, current, label):
        mechanism = _compat.get_electrode_mechanism("iclamp")
        mech = mechanism(tstart * U.ms, duration * U.ms, current * U.nA)
        _compat.place_current_source(decor, locset, mech, label)

    def __call__(self, i):
        tree = self._build_tree(i)
        decor, labels = self._build_decor(i)
        return {
            "tree": tree,
            "decor": decor,
            "labels": labels,
            "discretization": arbor.cv_policy_single(),
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
