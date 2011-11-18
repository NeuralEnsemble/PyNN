"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
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

# --- For implementation of record_X()/get_X()/print_X() -----------------------

def filter_variables(segment, variables):
    """
    Return a new `Segment` containing only recordings of the variables given in
    the list `variables`
    """
    if variables == 'all':
        return segment
    else:
        new_segment = copy(segment) # shallow copy
        if 'spikes' not in variables:
            new_segment.spiketrains = []
        new_segment.analogsignals = [sig for sig in segment.analogsignals if sig.name in variables]
        # also need to handle Units, RecordingChannels
        return new_segment


class DataCache(object):
    # primitive implementation for now, storing in memory - later can consider caching to disk
    def __init__(self):
        self._data = []

    def __iter__(self):
        return iter(self._data)
    
    def store(self, obj):
        self._data.append(obj)
        


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
    
    def __init__(self, population, file=None):
        __doc__ = super(Recorder, self).__init__.__doc__
        super(Recorder, self).__init__(population, file)
        self.cache = DataCache()
    
    def _record(self, variable, new_ids):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable == 'spikes':
            for id in new_ids:
                id._cell.record(1)
        elif variable == 'v':
            for id in new_ids:
                id._cell.record_v(1)
        elif variable == 'gsyn_exc':
            for id in new_ids:
                id._cell.record_gsyn("excitatory", 1)
                if id._cell.excitatory_TM is not None:
                    id._cell.record_gsyn("excitatory_TM", 1)
        elif variable == 'gsyn_inh':
             for id in new_ids:
                id._cell.record_gsyn("inhibitory", 1)
                if id._cell.inhibitory_TM is not None:
                    id._cell.record_gsyn("inhibitory_TM", 1)
        else:
            for id in new_ids:
               self._native_record(variable, id)
    
    def _reset(self):
        for id in self.recorded:
            id._cell.traces = {}
            id._cell.record(active=False)
            id._cell.record_v(active=False)
            for syn_name in id._cell.gsyn_trace:
                id._cell.record_gsyn(syn_name, active=False)
    
    def _native_record(self, variable, id):
        match = recordable_pattern.match(variable)
        if match:
            parts = match.groupdict()
            if parts['section']:
                section = getattr(id._cell, parts['section'])
                if parts['location']:
                    segment = section(float(parts['location']))
                else:
                    segment = section
            else:
                segment = id._cell.source
            id._cell.traces[variable] = vec = h.Vector()
            vec.record(getattr(segment, "_ref_%s" % parts['var']))
            if not id._cell.recording_time:
                id._cell.record_times = h.Vector()
                id._cell.record_times.record(h._ref_t)
                id._cell.recording_time += 1
        else:
            raise Exception("Recording of %s not implemented." % variable)
    
    def _get(self, variables, gather=False, filter_ids=None):
        """Return the recorded data as a Neo `Block`."""
        data = neo.Block()
        data.segments = [filter_variables(segment, variables) for segment in self.cache]
        data.segments.append(self._get_current_segment(filter_ids=filter_ids, variables=variables))
        if gather and simulator.state.num_processes > 1:
            data = recording.gather(data)
        return data
    
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
                    for id in self.filter_recorded('spikes', filter_ids)]
            else:
                if variable == 'v':
                    get_signal = lambda id: id._cell.vtrace
                elif variable == 'gsyn_exc':
                    get_signal = lambda id: id._cell.gsyn_trace['excitatory']
                elif variable == 'gsyn_inh':
                    get_signal = lambda id: id._cell.gsyn_trace['inhibitory']
                else:
                    signal = lambda id: id._cell.traces[variable]
                segment.analogsignals = [
                    neo.AnalogSignal(get_signal(id), # assuming not using cvode, otherwise need to use IrregularlySampledAnalogSignal
                                     units=recording.UNITS_MAP.get(variable, 'dimensionless'),
                                     t_start=simulator.state.t_start*pq.ms,
                                     sampling_period=simulator.state.dt*pq.ms,
                                     name=variable,
                                     source_population=self.population.label,
                                     source_id=int(id))
                    for id in self.filter_recorded(variable, filter_ids)
                ]
                assert segment.analogsignals[0].t_stop - simulator.state.t*pq.ms < simulator.state.dt*pq.ms
                # need to add `Unit` and `RecordingChannel` objects
        return segment
        
    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = id._cell.spike_times.size()
        else:
            raise Exception("Only implemented for spikes")
        return N
