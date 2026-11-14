"""

"""

from collections import defaultdict
import numpy as np
import arbor
from arbor import units as U

from .. import recording
from . import simulator, _compat
from ..morphology import LocationGenerator
from .morphology import LabelledLocations


class Recorder(recording.Recorder):
    _simulator = simulator

    def record(self, variables, ids, sampling_interval=None, locations=None):
        super().record(variables, ids, sampling_interval, locations)

    def _record(self, variable, new_ids, sampling_interval=None):
        if variable != "spikes" and sampling_interval is not None:
            self.sampling_interval = sampling_interval

    def _localize_variables(self, variables, locations):
        """

        """
        # If variables is a single string, encapsulate it in a list.
        if isinstance(variables, str) and variables != 'all':
            variables = [variables]
        if isinstance(locations, str):
            locations = [locations]
        resolved_variables = []

        if locations is None:
            for var_path in variables:
                resolved_variables.append(recording.Variable(location=None, name=var_path, label=None))
        else:
            if not isinstance(locations, (list, tuple)):
                assert isinstance(locations, (str, LocationGenerator))
                locations = [locations]

            for item in locations:
                if isinstance(item, str):
                    location_generator = LabelledLocations(item)
                elif isinstance(item, LocationGenerator):
                    location_generator = item
                else:
                    raise ValueError("'locations' should be a str, list, LocationGenerator or None")
                morphology = self.population.celltype.parameter_space["morphology"].base_value
                # todo: handle inhomogeneous morphologies in a Population
                for locset, label in location_generator.generate_locations(morphology, label="recording-point"):
                    short_label = label[len("recording-point-"):]
                    # not sure we need the 'recording-point' in the first place
                    for var_name in variables:
                        resolved_variables.append(recording.Variable(location=locset, name=var_name, label=short_label))
        return resolved_variables

    def _set_arbor_sim(self, arbor_sim):
        # Since Arbor 0.10.0, probes are addressed by (gid, tag) rather than by
        # a positional cell_member index. The tag assigned here (a per-gid probe
        # counter, in the same iteration order as _get_arbor_probes) must match
        # the tag given to the corresponding probe there.
        self.handles = defaultdict(dict)  # variable -> {gid: handle}
        probe_indices = defaultdict(int)
        for variable in self.recorded:
            if variable.name != "spikes":
                for cell in self.recorded[variable]:
                    tag = str(probe_indices[cell.gid])
                    probe_indices[cell.gid] += 1
                    handle = arbor_sim.sample(
                        cell.gid, tag, arbor.regular_schedule(self.sampling_interval * U.ms))
                    self.handles[variable][cell.gid] = handle

    def _get_arbor_probes(self, gid):
        probes = []
        probe_index = 0
        for variable in self.recorded:
            if variable.location is None:
                # Point neurons (single-compartment cable cells) are recorded
                # without an explicit location; default to the soma.
                locset = "(root)"
            else:
                locset = variable.location

            if gid in [cell.gid for cell in self.recorded[variable]]:
                if variable.name == "spikes":
                    continue
                # Tag must match the one assigned in _set_arbor_sim (per-gid index).
                tag = str(probe_index)
                probe_index += 1
                if variable.name == "v":
                    if self.population.celltype.arbor_cell_kind == arbor.cell_kind.lif:
                        # Native lif cells have no morphology/locset.
                        probe = arbor.lif_probe_voltage(tag)
                    else:
                        probe = arbor.cable_probe_membrane_voltage(locset, tag)
                else:
                    probe = self._point_mechanism_probe(gid, variable.name, tag)
                    if probe is None:
                        mech_name, state_name = variable.name.split(".")
                        arbor_model = mech_name  # to do: find_arbor_model(mech_name)
                        probe = arbor.cable_probe_density_state(locset, arbor_model, state_name, tag)
                probes.append(probe)
        return probes

    # standard-model names for post-synaptic receptor variables
    _receptor_variables = {
        "gsyn_exc": ("excitatory", "gsyn"),
        "gsyn_inh": ("inhibitory", "gsyn"),
        "isyn_exc": ("excitatory", "isyn"),
        "isyn_inh": ("inhibitory", "isyn"),
    }
    # state variable of the synapse mechanism holding each quantity
    _synaptic_states = {"gsyn": "g", "isyn": "isyn"}

    def _point_mechanism_probe(self, gid, variable_name, tag):
        """
        Return a probe for a point-neuron state variable held by a point mechanism,
        or None if ``variable_name`` is not one of these. This covers state variables
        of the neuron's reset/dynamics process (e.g. "w", "u") and synaptic
        conductances or currents, given either as a standard name (e.g. "gsyn_exc")
        or as "<receptor>.<gsyn|isyn>".
        """
        builder = getattr(self.population._arbor_cell_description, "base_value", None)
        receptors = getattr(builder, "post_synaptic_receptors", None)
        if receptors is None:
            return None
        if variable_name in builder.dynamics.point_states:
            target, model, state = builder.dynamics.point_states[variable_name]
        else:
            if variable_name in self._receptor_variables:
                target, quantity = self._receptor_variables[variable_name]
            elif variable_name.count(".") == 1:
                target, quantity = variable_name.split(".")
            else:
                return None
            if target not in receptors or quantity not in self._synaptic_states:
                return None
            model, state = receptors[target][0], self._synaptic_states[quantity]
        index = self.population.id_to_index(gid)
        return _compat.point_state_probe(lambda: builder(index)["decor"], target, model, state, tag)

    def _get_spiketimes(self, id, clear=False):
        spikes = self._simulator.state.arbor_sim.spikes()
        # filter to keep only the spikes from the requested ids
        mask = np.isin(spikes["source"]["gid"], id)
        my_spikes = spikes[mask]
        return my_spikes["source"]["gid"].ravel(), my_spikes["time"].ravel()

    def _get_all_signals(self, variable, ids, clear=False):
        all_data = []
        # signals must be returned in the order of ``ids``, which the caller has
        # already restricted to the requested (e.g. PopulationView) cells
        for id in ids:
            handle = self.handles[variable][int(id)]
            data = []
            meta = []
            for d, m in self._simulator.state.arbor_sim.samples(handle):
                data.append(d[:, 1])
                meta.append(m)
            if not data:
                # Arbor returns no samples when a probe matches nothing on the cell
                raise NotImplementedError(
                    f"pyNN.arbor cannot record '{variable.name}' from cell type "
                    f"{self.population.celltype.__class__.__name__}")
            all_data.extend(data)
        if not all_data:
            return np.zeros((0, 0)), None
        signal_array = np.stack(all_data, axis=1)
        times_array = None
        return signal_array, times_array

    def _local_count(self, variable, filter_ids=None):
        if variable.name != "spikes":
            raise Exception("Only implemented for spikes")
        gids = self._simulator.state.arbor_sim.spikes()["source"]["gid"].ravel()
        counts = np.bincount(gids.astype(int)) if gids.size else np.zeros(0, dtype=int)
        return {
            int(id): int(counts[id]) if id < counts.size else 0
            for id in self.filter_recorded(variable, filter_ids)
        }

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
