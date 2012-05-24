"""

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from datetime import datetime
from pyNN import recording
from pyNN.neuron import simulator
import re
from neuron import h
import neo
import quantities as pq
from copy import copy

recordable_pattern = re.compile(r'((?P<section>\w+)(\((?P<location>[-+]?[0-9]*\.?[0-9]+)\))?\.)?(?P<var>\w+)')


def find_units(variable):
    if variable in recording.UNITS_MAP:
        return recording.UNITS_MAP[variable]
    else:
        # works with NEURON 7.3, not with 7.1, 7.2 not tested
        nrn_units = h.units(variable.split('.')[-1])
        pq_units = nrn_units.replace("2", "**2").replace("3", "**3")
        return pq_units


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def _record(self, variable, new_ids):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable == 'spikes':
            for id in new_ids:
                id._cell.rec.record(id._cell.spike_times)
        else:
            for id in new_ids:
                self._record_state_variable(id._cell, variable)

    def _record_state_variable(self, cell, variable):
        if variable == 'v':
            hoc_var = cell(0.5)._ref_v  # or use "seg.v"?
        elif variable == 'gsyn_exc':
            if cell.excitatory_TM is None:
                hoc_var = cell.esyn._ref_g
            else:
                hoc_var = cell.esyn_TM._ref_g
        elif variable == 'gsyn_inh':
            if cell.inhibitory_TM is None:
                hoc_var = cell.isyn._ref_g
            else:
                hoc_var = cell.isyn_TM._ref_g
        else:
            source, var_name = self._resolve_variable(cell, variable)
            hoc_var = getattr(source, "_ref_%s" % var_name)
        cell.traces[variable] = vec = h.Vector()
        vec.record(hoc_var)
        if not cell.recording_time:
            cell.record_times = h.Vector()
            cell.record_times.record(h._ref_t)
            cell.recording_time += 1

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
        for id in set.union(*self.recorded.values()):
            id._cell.traces = {}
            id._cell.spike_times = h.Vector(0)
        id._cell.recording_time == 0
        id._cell.record_times = None

    def _get_current_segment(self, filter_ids=None, variables='all'):
        segment = neo.Segment(name=self.population.label,
                              description=self.population.describe(),
                              rec_datetime=datetime.now()) # would be nice to get the time at the start of the recording, not the end
        variables_to_include = set(self.recorded.keys())
        if variables is not 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        def trim_spikes(spikes):
            return spikes[spikes<=simulator.state.t+1e-9]
        #import pdb; pdb.set_trace()
        for variable in variables_to_include:
            if variable == 'spikes':
                segment.spiketrains = [
                    neo.SpikeTrain(trim_spikes(numpy.array(id._cell.spike_times)),
                                   t_stop=simulator.state.t*pq.ms,
                                   units='ms',
                                   source_population=self.population.label,
                                   source_id=int(id)) # index?
                    for id in sorted(self.filter_recorded('spikes', filter_ids))]
            else:
                ids = sorted(self.filter_recorded(variable, filter_ids))
                signal_array = numpy.vstack((id._cell.traces[variable] for id in ids))
                channel_indices = [self.population.id_to_index(id) for id in ids]
                segment.analogsignalarrays.append(
                    neo.AnalogSignalArray(
                        signal_array.T, # assuming not using cvode, otherwise need to use IrregularlySampledAnalogSignal
                        units=find_units(variable),
                        t_start=simulator.state.t_start*pq.ms,
                        sampling_period=simulator.state.dt*pq.ms,
                        name=variable,
                        source_population=self.population.label,
                        channel_indexes=channel_indices,
                        source_ids=numpy.fromiter(ids, dtype=int))
                )
                assert segment.analogsignalarrays[0].t_stop - simulator.state.t*pq.ms < 2*simulator.state.dt*pq.ms
                # need to add `Unit` and `RecordingChannelGroup` objects
        return segment

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes")
        return N
