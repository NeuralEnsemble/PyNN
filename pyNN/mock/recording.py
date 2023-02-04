import numpy as np
from .. import recording
from . import simulator


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        pass

    def _get_spiketimes(self, id, clear=False):
        if hasattr(id, "__len__"):
            spks = {}
            for i in id:
                spks[i] = np.array([i, i + 5], dtype=float) % self._simulator.state.t
            return spks
        else:
            return np.array([id, id + 5], dtype=float) % self._simulator.state.t

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well
        # and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t / self._simulator.state.dt)) + 1
        return np.vstack([np.random.uniform(size=n_samples) for id in ids]).T, None

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
