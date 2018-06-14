import numpy
from pyNN import recording
from . import simulator


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        pass

    def _get_spiketimes(self, id_):
        # common.Recorder._get_current_segment creates id_ as a set, so we must
        # cast it to numpy.array if we want to add a scalar to each element
        # below
        id_ = numpy.array(id_)
        data = numpy.array([id_, id_ + 5], dtype=float) % self._simulator.state.t

        # _get_spiketimes is expected to return a dict
        retval = {x[0]: x[1] for x in data}
        print(retval)
        return retval

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use
        # IrregularlySampledAnalogSignal
        n_samples = (int(round(self._simulator.state.t /
                               self._simulator.state.dt)) + 1)
        return (
            numpy.vstack((
                numpy.random.uniform(size=n_samples) for id_ in ids)).T
        )

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id_ in self.filter_recorded(variable, filter_ids):
                N[int(id_)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
