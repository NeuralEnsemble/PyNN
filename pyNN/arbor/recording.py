"""

"""

from collections import defaultdict
import numpy as np
import arbor

from .. import recording
from . import simulator


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        if variable != "spikes" and sampling_interval is not None:
            self.sampling_interval = sampling_interval

    def _set_arbor_sim(self, arbor_sim):
        self.handles = defaultdict(list)
        probe_indices = defaultdict(int)
        for variable in self.recorded:
            if variable.name != "spikes":
                for cell in self.recorded[variable]:
                    probeset_id = arbor.cell_member(cell.gid, probe_indices[cell.gid])
                    probe_indices[cell.gid] += 1
                    handle = arbor_sim.sample(probeset_id, arbor.regular_schedule(self.sampling_interval))
                    self.handles[variable].append(handle)

    def _get_arbor_probes(self, gid):
        probes = []
        for variable in self.recorded:
            locset = f'(on-components 0.5 (region "{variable.location}"))'  # or variable.location_label?
            if gid in [cell.gid for cell in self.recorded[variable]]:
                if variable.name == "spikes":
                    continue
                elif variable.name == "v":
                    probe = arbor.cable_probe_membrane_voltage(locset)
                else:
                    mech_name, state_name = variable.name.split(".")
                    arbor_model = "hh"  # to do: find_arbor_model(mech_name)
                    probe = arbor.cable_probe_density_state(locset, arbor_model, state_name)
                probes.append(probe)
        return probes

    def _get_spiketimes(self, id, clear=False):
        spikes = self._simulator.state.arbor_sim.spikes()
        # filter to keep only the spikes from the requested ids
        mask = np.isin(spikes["source"]["gid"], id)
        my_spikes = spikes[mask]
        return my_spikes["source"]["gid"].ravel(), my_spikes["time"].ravel()

    def _get_all_signals(self, variable, ids, clear=False):
        all_data = []
        for handle in self.handles[variable]:
            data = []
            meta = []
            for d, m in self._simulator.state.arbor_sim.samples(handle):
                data.append(d[:, 1])
                meta.append(m)
            all_data.extend(data)
        signal_array = np.stack(all_data, axis=1)
        times_array = None
        return signal_array, times_array

    def _local_count(self, variable, filter_ids=None):
        raise NotImplementedError()

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
