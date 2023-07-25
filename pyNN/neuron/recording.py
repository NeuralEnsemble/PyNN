"""

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from collections import defaultdict
import numpy as np
from .. import recording
from ..morphology import MorphologyFilter
from . import simulator
import re
from neuron import h


recordable_pattern = re.compile(
    r'((?P<section>\w+)(\((?P<location>[-+]?[0-9]*\.?[0-9]+)\))?\.)?(?P<var>\w+)')


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable.name == 'spikes':
            for id in new_ids:
                if id._cell.rec is not None:
                    id._cell.rec.record(id._cell.spike_times)
                else:  # SpikeSourceArray
                    id._cell.recording = True
        else:
            self.sampling_interval = sampling_interval or self._simulator.state.dt
            for id in new_ids:
                self._record_state_variable(id._cell, variable)

    def _record_state_variable(self, cell, variable):
        if variable.location is None:
            if hasattr(cell, 'recordable') and variable in cell.recordable:
                hoc_var = cell.recordable[variable]
            elif variable.name == 'v':
                hoc_var = cell.source_section(0.5)._ref_v  # or use "seg.v"?
            elif variable.name == 'gsyn_exc':
                hoc_var = cell.esyn._ref_g
            elif variable.name == 'gsyn_inh':
                hoc_var = cell.isyn._ref_g
            else:
                source, var_name = self._resolve_variable(cell, variable.name)
                if hasattr(self.population.celltype, "variable_map"):
                    var_name = self.population.celltype.variable_map[variable.name]
                hoc_var = getattr(source, "_ref_%s" % var_name)
            hoc_vars = [hoc_var]
        else:
            if isinstance(variable.location, str):
                if variable.location in cell.section_labels:
                    section_index = cell.section_labels[variable.location]
                else:
                    raise ValueError("Cell has no location labelled '{}'".format(variable.location))
            elif isinstance(variable.location, MorphologyFilter):
                section_index = variable.location(cell.morphology)
            else:
                raise ValueError("Invalid location specification: {}".format(variable.location))
            if hasattr(section_index, "__len__"):
                sections = [cell.sections[index] for index in section_index]
            else:
                sections = [cell.sections[section_index]]
            hoc_vars = []
            for section in sections:
                source = section(0.5)
                if variable.name == 'v':
                    hoc_vars.append(source._ref_v)
                else:
                    ion_channel, var_name = variable.name.split(".")
                    mechanism_name, hoc_var_name = self.population.celltype.ion_channels[ion_channel].variable_translations[var_name]
                    mechanism = getattr(source, mechanism_name)
                    hoc_vars.append(getattr(mechanism, "_ref_{}".format(hoc_var_name)))
        for hoc_var in hoc_vars:
            vec = h.Vector()
            if self.sampling_interval == self._simulator.state.dt or self.record_times:
                vec.record(hoc_var)
            else:
                vec.record(hoc_var, self.sampling_interval)
            cell.traces[variable].append(vec)
        if not cell.recording_time:
            cell.recorded_times = h.Vector()
            if self.sampling_interval == self._simulator.state.dt or self.record_times:
                cell.recorded_times.record(h._ref_t)
            else:
                cell.recorded_times.record(h._ref_t, self.sampling_interval)
            cell.recording_time += 1

    # could be staticmethod
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
            id._cell.traces = defaultdict(list)
            id._cell.spike_times = h.Vector(0)
        id._cell.recording_time == 0
        id._cell.recorded_times = None

    def _clear_simulator(self):
        """
        Should remove all recorded data held by the simulator and, ideally,
        free up the memory.
        """
        for id in set.union(*self.recorded.values()):
            if hasattr(id._cell, "traces"):
                for variable in id._cell.traces:
                    for vec in id._cell.traces[variable]:
                        vec.resize(0)
            if id._cell.rec is not None:
                id._cell.spike_times.resize(0)
            else:
                id._cell.clear_past_spikes()

    def _get_spiketimes(self, id, clear=False):
        if hasattr(id, "__len__"):
            all_spiketimes = {}
            for cell_id in id:
                if cell_id._cell.rec is None:  # SpikeSourceArray
                    spikes = cell_id._cell.get_recorded_spike_times()
                else:
                    spikes = cell_id._cell.spike_times.as_numpy()
                all_spiketimes[cell_id] = spikes[spikes <= simulator.state.t + 1e-9]
            return all_spiketimes
        else:
            spikes = id._cell.spike_times.as_numpy()
            return spikes[spikes <= simulator.state.t + 1e-9]

    def _get_all_signals(self, variable, ids, clear=False):
        times = None
        if len(ids) > 0:
            # note: id._cell.traces[variable] is a list of Vectors, one per segment
            signals = np.vstack((vec for id in ids for vec in id._cell.traces[variable])).T
            if self.record_times:
                assert not simulator.state.cvode.use_local_dt()
                # the following line assumes all cells are sampled at the same time
                # which should be true if cvode.use_local_dt() returns False
                times = np.array(ids[0]._cell.recorded_times)
            else:
                expected_length = np.rint(simulator.state.tstop / self.sampling_interval) + 1
                if signals.shape[0] != expected_length:
                    # generally due to floating point/rounding issues
                    signals = np.vstack((signals, signals[-1, :]))
                if ".isyn" in variable:
                    # this is a hack, since negative currents in NMODL files
                    # correspond to positive currents in PyNN
                    # todo: reimplement this in a more robust way
                    signals *= -1
        else:
            signals = np.array([])
        return signals, times

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable.name == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes")
        return N
