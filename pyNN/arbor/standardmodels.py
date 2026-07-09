"""
Standard cells for the Arbor module.

:copyright: Copyright 2006-2026 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from copy import deepcopy

import numpy as np
import arbor
from arbor import units as U

from ..standardmodels import cells, ion_channels, synapses, electrodes, receptors, build_translations
from ..parameters import ParameterSpace, IonicSpecies, Sequence
from ..morphology import Morphology, NeuriteDistribution, LocationGenerator
from .cells import CellDescriptionBuilder, PointCellDescriptionBuilder
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
    arbor_schedule_units = {"tstart": U.ms, "freq": U.Hz, "tstop": U.ms}


class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'times'),
    )
    arbor_cell_kind = arbor.cell_kind.spike_source
    arbor_schedule = arbor.explicit_schedule
    # Since Arbor 0.10.0, schedule parameters must be unit-typed.
    arbor_schedule_units = {"times": U.ms}


class IF_curr_delta(cells.IF_curr_delta):
    __doc__ = cells.IF_curr_delta.__doc__

    # Maps onto Arbor's native leaky integrate-and-fire cell (arbor.lif_cell,
    # cell_kind.lif). Its synapses are delta, but an incoming event adds
    # weight/C_m to V_m (the event weight is a charge, not a voltage), so
    # IF_curr_delta's mV voltage-step weight is recovered by scaling the
    # connection weight by C_m (see Projection._lif_post_cm_pF).
    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'E_R'),
        ('v_thresh',   'V_th'),
        ('tau_refrac', 't_ref'),
        ('tau_m',      'tau_m'),
        ('cm',         'C_m'),
        # A native lif_cell has no way to inject a constant current, so i_offset
        # is carried through untranslated and rejected at cell-build time if
        # non-zero (see Population.arbor_cell_description).
        ('i_offset',   'i_offset'),
    )
    arbor_cell_kind = arbor.cell_kind.lif
    # Units for the native lif_cell attributes (Arbor requires unit-typed values,
    # and handles the conversion to its internal units itself).
    lif_param_units = {
        'E_L': U.mV, 'E_R': U.mV, 'V_th': U.mV,
        't_ref': U.ms, 'tau_m': U.ms, 'C_m': U.nF,
    }


# --- current sources ---------------------------------------------------------
#
# Every standard current source is realised as one or more Arbor iclamp
# "components", each a (envelope, frequency_Hz, phase_deg) tuple where ``envelope``
# is a list of (time_ms, amplitude_nA) points (see cells.BaseCellDescriptionBuilder).
# Arbor iclamp envelope semantics: the current is 0 before the first point, held at
# the last amplitude after the last point, linearly interpolated between points, and
# steps discontinuously where two points share a time. With a non-zero frequency the
# envelope amplitude-modulates a sine, sin(2*pi*f*t + phase), referenced to t=0.

_CURRENT_STOP_SENTINEL = 1e11  # PyNN's "run to the end" stop default is 1e12


def _as_array(value):
    """Coerce a (possibly Sequence-wrapped) parameter to a float ndarray."""
    return np.asarray(value.value if isinstance(value, Sequence) else value, dtype=float)


def _box_envelope(start, stop, amplitude):
    """A rectangular pulse of ``amplitude`` over [start, stop], zero outside."""
    return [(start, amplitude), (stop, amplitude), (stop, 0.0)]


def _staircase_envelope(times, amplitudes):
    """A piecewise-constant staircase: ``amplitudes[k]`` is held from ``times[k]``
    until ``times[k+1]`` (and the last value until the end of the run). The current
    is zero before ``times[0]``. Duplicated timestamps make the steps discontinuous."""
    envelope = []
    for k in range(len(times)):
        if k > 0:
            envelope.append((times[k], amplitudes[k - 1]))
        envelope.append((times[k], amplitudes[k]))
    return envelope


def _check_step_times(times):
    """Validate StepCurrentSource times (mirrors the checks in the NEURON backend)."""
    if not (times >= 0.0).all():
        raise ValueError("Step current cannot accept negative timestamps.")
    if not (np.diff(times) > 0.0).all():
        raise ValueError("Step current timestamps should be monotonically increasing.")


class BaseCurrentSource(object):
    """Base class for the Arbor current sources.

    Subclasses implement :meth:`_iclamp_components`, returning the list of
    (envelope, frequency_Hz, phase_deg) components for the injected current; this
    base handles resolving the injection target and registering the components on
    the target cells' description builder.
    """

    def inject_into(self, cells, location=None):  # rename to `locations` ?
        if hasattr(cells, "parent"):
            target_pop = cells.parent
            cell_descr = target_pop._arbor_cell_description.base_value
            index = target_pop.id_to_index(cells.all_cells.astype(int))
        elif hasattr(cells, "_arbor_cell_description"):
            target_pop = cells
            cell_descr = cells._arbor_cell_description.base_value
            index = cells.id_to_index(cells.all_cells.astype(int))
        else:
            assert isinstance(cells, (list, tuple))
            # we're assuming all cells have the same parent here
            target_pop = cells[0].parent
            cell_descr = target_pop._arbor_cell_description.base_value
            index = np.array(cells, dtype=int)

        # Native lif cells (IF_curr_delta) have no decor and cannot take an
        # i_clamp, so current injection is impossible for them.
        if target_pop.celltype.arbor_cell_kind == arbor.cell_kind.lif:
            raise NotImplementedError(
                "Current injection into Arbor's native lif_cell (IF_curr_delta) "
                "is not supported; use the cable-cell IF models instead.")

        self.parameter_space.shape = (1,)
        if location is None:
            # Point neurons (and, by default, any cell) inject at the soma.
            location = LabelledLocations("soma")
        elif isinstance(location, str):
            location = LabelledLocations(location)
        elif isinstance(location, LocationGenerator):
            pass
        else:
            raise TypeError("location must be a string or a LocationGenerator")

        cell_descr.add_current_source(
            components=self._iclamp_components(),
            location_generator=location,
            index=index,
        )

    def _native_parameters(self):
        """The source's native parameters as a plain {name: scalar} dict."""
        native = self.native_parameters
        native.shape = (1,)
        native.evaluate(simplify=True)
        return native.as_dict()

    def _iclamp_components(self):
        raise NotImplementedError("Should be redefined in the individual current sources")


class DCSource(BaseCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'current'),
        ('start',      'tstart'),
        ('stop',       'duration', "stop - start", "tstart + duration")
    )

    def _iclamp_components(self):
        p = self._native_parameters()
        start = p["tstart"]
        return [(_box_envelope(start, start + p["duration"], p["current"]), 0.0, 0.0)]


class StepCurrentSource(BaseCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )

    def _iclamp_components(self):
        p = self._native_parameters()
        times = _as_array(p["times"])
        amplitudes = _as_array(p["amplitudes"])
        _check_step_times(times)
        return [(_staircase_envelope(times, amplitudes), 0.0, 0.0)]


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

    def _iclamp_components(self):
        p = self._native_parameters()
        start, stop = p["start"], p["stop"]
        # Arbor references the sine to t=0, so shift the phase to make PyNN's
        # ``phase`` hold at ``start`` instead.
        phase = p["phase"] - 360.0 * p["frequency"] * start / 1000.0
        components = [(_box_envelope(start, stop, p["amplitude"]), p["frequency"], phase)]
        if p["offset"] != 0.0:
            # Arbor sums co-located clamps, so the DC offset is a separate clamp.
            components.append((_box_envelope(start, stop, p["offset"]), 0.0, 0.0))
        return components


class NoisyCurrentSource(BaseCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__

    translations = build_translations(
        ('mean',  'mean'),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'stdev'),
        ('dt',    'dt')
    )

    def _iclamp_components(self):
        p = self._native_parameters()
        start, stop = p["start"], p["stop"]
        if stop >= _CURRENT_STOP_SENTINEL:
            raise ValueError(
                "NoisyCurrentSource on the Arbor backend must be given a finite `stop` "
                "(the noise is precomputed as a per-sample current envelope).")
        dt = max(p["dt"], state.dt)
        n = int(round((stop - start) / dt))
        times = np.append(start + dt * np.arange(n), stop)
        amplitudes = p["mean"] + p["stdev"] * np.random.randn(len(times))
        amplitudes[-1] = 0.0  # switch the current off at `stop`
        return [(_staircase_envelope(times, amplitudes), 0.0, 0.0)]


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
        # Instantiate the ion-channel and post-synaptic classes into new,
        # instance-level dicts. These are declared as class attributes holding
        # the classes; building fresh dicts here (rather than assigning into the
        # class-level dict) avoids mutating that shared state -- otherwise a
        # second instantiation would try to call an already-instantiated object
        # ("'PassiveLeak' object is not callable").
        self.ion_channels = {
            name: ion_channel(**parameters.pop(name, {}))
            for name, ion_channel in self.ion_channels.items()
        }
        self.post_synaptic_entities = {
            name: pse(**parameters.pop(name, {}))
            for name, pse in self.post_synaptic_entities.items()
        }
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
        return schema

    def translate(self, parameters, copy=True):
        """Translate standardized model parameters to simulator-specific parameters."""
        if copy:
            _parameters = deepcopy(parameters)
        else:
            _parameters = parameters
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


class CurrExpPostSynapticResponse(receptors.CurrExpPostSynapticResponse):

    translations = build_translations(
        ('locations', 'locations'),
        ('tau_syn', 'tau')
    )
    model = "expsyn_curr"
    recordable = ["isyn"]
    variable_map = {"isyn": "isyn"}


class LIF(cells.LIF):
    __doc__ = cells.LIF.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'E_R'),
        ('v_thresh',   'V_th'),
        ('tau_refrac', 't_ref'),
        ('tau_m',      'tau_m'),
        ('cm',         'C_m'),
        ('i_offset',   'i_offset'),
    )
    variable_map = {"v": "v"}


class PointNeuron(cells.PointNeuron):
    """Composable point neuron, realised as a single-compartment Arbor cable cell.

    Combines a leaky integrate-and-fire ``neuron`` (an :class:`LIF` instance) with
    one or more post-synaptic receptors (:class:`CondExpPostSynapticResponse` or
    :class:`CurrExpPostSynapticResponse`). See :class:`PointCellDescriptionBuilder`
    for how the cable cell is assembled.
    """

    arbor_cell_kind = arbor.cell_kind.cable

    def translate(self, parameters, copy=True):
        """Build the Arbor cable-cell description for this point neuron.

        ``parameters`` (the composable parameter space) is not consumed directly:
        the neuron and receptor components carry their own (translatable) parameter
        spaces, which are assembled into the form the builder expects.
        """
        neuron_parameters = self.neuron.native_parameters
        post_synaptic_receptors = {
            name: (psr.model, psr.native_parameters)
            for name, psr in self.post_synaptic_receptors.items()
        }
        builder = PointCellDescriptionBuilder(neuron_parameters, post_synaptic_receptors)
        return ParameterSpace({"cell_description": builder}, schema=None, shape=parameters.shape)

    def reverse_translate(self, native_parameters):
        raise NotImplementedError

    def can_record(self, variable, location=None):
        return True  # todo: implement this properly


# The native (LIF) parameter names carried through to PointCellDescriptionBuilder;
# the remaining native names produced by the classic IF models below describe their
# synapses.
_LIF_NATIVE_NAMES = ("E_L", "E_R", "V_th", "t_ref", "tau_m", "C_m", "i_offset")


def _point_cell_description(native, receptor_specs, shape):
    """Wrap a flat native parameter space (from a classic IF model's base
    ``translate()``) into a point-neuron ``cell_description`` ParameterSpace.

    ``receptor_specs`` maps each receptor label to
    ``(arbor_synapse_model, {arbor_synapse_param: native_name})``.
    """
    neuron_parameters = ParameterSpace(
        {name: native[name] for name in _LIF_NATIVE_NAMES}, shape=shape)
    post_synaptic_receptors = {
        label: (model,
                ParameterSpace({arbor_param: native[native_name]
                                for arbor_param, native_name in param_map.items()},
                               shape=shape))
        for label, (model, param_map) in receptor_specs.items()
    }
    builder = PointCellDescriptionBuilder(neuron_parameters, post_synaptic_receptors)
    return ParameterSpace({"cell_description": builder}, schema=None, shape=shape)


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'E_R'),
        ('v_thresh',   'V_th'),
        ('tau_refrac', 't_ref'),
        ('tau_m',      'tau_m'),
        ('cm',         'C_m'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
    )
    arbor_cell_kind = arbor.cell_kind.cable

    def translate(self, parameters, copy=True):
        native = super().translate(parameters, copy)
        return _point_cell_description(native, {
            "excitatory": ("expsyn_curr", {"tau": "tau_syn_E"}),
            "inhibitory": ("expsyn_curr", {"tau": "tau_syn_I"}),
        }, parameters.shape)


class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('v_rest',     'E_L'),
        ('v_reset',    'E_R'),
        ('v_thresh',   'V_th'),
        ('tau_refrac', 't_ref'),
        ('tau_m',      'tau_m'),
        ('cm',         'C_m'),
        ('i_offset',   'i_offset'),
        ('tau_syn_E',  'tau_syn_E'),
        ('tau_syn_I',  'tau_syn_I'),
        ('e_rev_E',    'e_rev_E'),
        ('e_rev_I',    'e_rev_I'),
    )
    arbor_cell_kind = arbor.cell_kind.cable

    def translate(self, parameters, copy=True):
        native = super().translate(parameters, copy)
        return _point_cell_description(native, {
            "excitatory": ("expsyn", {"tau": "tau_syn_E", "e": "e_rev_E"}),
            "inhibitory": ("expsyn", {"tau": "tau_syn_I", "e": "e_rev_I"}),
        }, parameters.shape)
