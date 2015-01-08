"""

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from datetime import datetime
from pyNN import recording
from pyNN.neuron import simulator
import re
from neuron import h
import neo
from copy import copy

recordable_pattern = re.compile(r'((?P<section>\w+)(\((?P<location>[-+]?[0-9]*\.?[0-9]+)\))?\.)?(?P<var>\w+)')


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable == 'spikes':
            for id in new_ids:
                if id._cell.rec is not None:
                    id._cell.rec.record(id._cell.spike_times)
        else:
            self.sampling_interval = sampling_interval or self._simulator.state.dt
            for id in new_ids:
                self._record_state_variable(id._cell, variable)

    def _record_state_variable(self, cell, variable):
        if hasattr(cell, 'recordable') and variable in cell.recordable:
            hoc_var = cell.recordable[variable]
        elif variable == 'v':
            hoc_var = cell.source_section(0.5)._ref_v  # or use "seg.v"?
        elif variable == 'gsyn_exc':
            hoc_var = cell.esyn._ref_g
        elif variable == 'gsyn_inh':
            hoc_var = cell.isyn._ref_g
        else:
            source, var_name = self._resolve_variable(cell, variable)
            hoc_var = getattr(source, "_ref_%s" % var_name)
        cell.traces[variable] = vec = h.Vector()
        vec.record(hoc_var, self.sampling_interval)
        if not cell.recording_time:
            cell.record_times = h.Vector()
            cell.record_times.record(h._ref_t, self.sampling_interval)
            cell.recording_time += 1

    #could be staticmethod
    def _resolve_variable(self, cell, variable_path):
        match = recordable_pattern.match(variable_path)
        if match:
            parts = match.groupdict()
            if parts['section']:
                section = getattr(cell, parts['section'])
                if parts['location']:
                    source = section(float(parts['location']))
                else:
                    source = section
            else:
                source = cell.source
            return source, parts['var']
        else:
            raise AttributeError("Recording of %s not implemented." % variable_path)

    def _reset(self):
        """Reset the list of things to be recorded."""
        for id in set.union(*self.recorded.values()):
            id._cell.traces = {}
            id._cell.spike_times = h.Vector(0)
        id._cell.recording_time == 0
        id._cell.record_times = None

    def _clear_simulator(self):
        """
        Should remove all recorded data held by the simulator and, ideally,
        free up the memory.
        """
        for id in set.union(*self.recorded.values()):
            if hasattr(id._cell, "traces"):
                for variable in id._cell.traces:
                    id._cell.traces[variable].resize(0)
            if id._cell.rec is not None:
                id._cell.spike_times.resize(0)
            else:
                id._cell.clear_past_spikes()

    def _get_spiketimes(self, id):
        spikes = numpy.array(id._cell.spike_times)
        return spikes[spikes <= simulator.state.t + 1e-9]

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        if len(ids) > 0:
            signals = numpy.vstack((id._cell.traces[variable] for id in ids)).T
            expected_length = int(simulator.state.tstop/self.sampling_interval) + 1
            if signals.shape[0] != expected_length:  # generally due to floating point/rounding issues
                signals = numpy.vstack((signals, signals[-1, :]))
        else:
            signals = numpy.array([])
        return signals

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes")
        return N
