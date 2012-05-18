"""

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import brian
from pyNN import recording
from pyNN.brian import simulator
import logging
import neo
import quantities as pq
from datetime import datetime

mV = brian.mV
ms = brian.ms
uS = brian.uS

logger = logging.getLogger("PyNN")

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def __init__(self, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, population, file)
        self._devices = {} # defer creation until first call of record()

    def _create_device(self, group, variable):
        """Create a Brian recording device."""
        clock = simulator.state.simclock
        if variable == 'spikes':
            self._devices[variable] = brian.SpikeMonitor(group, record=self.recorded)
        else:
            if variable == 'v':
                varname = 'v'
            elif variable == 'gsyn_exc':
                varname = self.population.celltype.synapses['excitatory']
            elif variable == 'gsyn_inh':
                varname = self.population.celltype.synapses['inhibitory']
            self._devices[variable] = brian.StateMonitor(group, varname,
                                                   record=self.recorded, clock=clock)
        simulator.state.add(self._devices[variable])

    def _record(self, variable, new_ids):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if variable not in self._devices:
            self._create_device(self.population.brian_cells, variable)
        #update StateMonitor.record and StateMonitor.recordindex
        if not variable is 'spikes':
            device = self._devices[variable]
            device.record = numpy.fromiter(self.recorded[variable], dtype=int) - self.population.first_id
            device.recordindex = dict((i,j) for i,j in zip(device.record,
                                                           range(len(device.record))))
            logger.debug("recording %s from %s" % (variable, self.recorded[variable]))

    def _reset(self):
        raise NotImplementedError("Recording reset is not currently supported for pyNN.brian")

    def _get_current_segment(self, filter_ids=None, variables='all'):
        segment = neo.Segment(name=self.population.label,
                              description=self.population.describe(),
                              rec_datetime=datetime.now()) # would be nice to get the time at the start of the recording, not the end
        variables_to_include = set(self.recorded.keys())
        if variables is not 'all':
            variables_to_include = variables_to_include.intersection(set(variables))
        padding = self.population.first_id

        for variable in variables_to_include:
            filtered_ids = self.filter_recorded(variable, filter_ids)
            indices = numpy.fromiter(filtered_ids, dtype=int) - padding
            if variable == 'spikes':
                spike_times = self._devices['spikes'].spiketimes
                t_stop = simulator.state.t*pq.ms # must run on all MPI nodes
                segment.spiketrains = [
                    neo.SpikeTrain(spike_times[i]/ms,
                                   t_stop=t_stop,
                                   units='ms',
                                   source_population=self.population.label,
                                   source_id=int(i + padding)) # index?
                    for i in indices]
            else:
                signal_array = numpy.array(self._devices[variable]._values)/mV # TODO detect & scale units properly
                t_start=simulator.state.t_start*pq.ms
                sampling_period=simulator.state.dt*pq.ms # must run on all MPI nodes
                segment.analogsignalarrays.append(
                    neo.AnalogSignalArray(
                        signal_array.T,
                        units=recording.UNITS_MAP.get(variable, 'dimensionless'),
                        t_start=t_start,
                        sampling_period=sampling_period,
                        name=variable,
                        source_population=self.population.label,
                        channel_indexes=indices,
                        source_ids=indices+padding)
                )
        return segment

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding      = self.population.first_id
        filtered_ids = numpy.fromiter(filtered_ids, dtype=int) - padding
        for id in filtered_ids:
            N[id] = len(self._devices['spikes'].spiketimes[id])
        return N
