"""

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import numpy
import quantities as pq
import neo
import brian
from pyNN import recording
from . import simulator

mV = brian.mV
ms = brian.ms
uS = brian.uS
pq.uS = pq.UnitQuantity('microsiemens', 1e-6*pq.S, 'uS')
pq.nS = pq.UnitQuantity('nanosiemens', 1e-9*pq.S, 'nS')

logger = logging.getLogger("PyNN")


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def __init__(self, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, population, file)
        self._devices = {} # defer creation until first call of record()

    def _create_device(self, group, variable):
        """Create a Brian recording device."""
        # By default, StateMonitor has when='end', i.e. the value recorded at
        # the end of the timestep is associated with the time at the start of the step,
        # This is different to the PyNN semantics (i.e. the value at the end of
        # the step is associated with the time at the end of the step.)
        clock = simulator.state.network.clock
        if variable == 'spikes':
            self._devices[variable] = brian.SpikeMonitor(group, record=self.recorded)
        else:
            varname = self.population.celltype.state_variable_translations[variable]['translated_name']
            self._devices[variable] = brian.StateMonitor(group, varname,
                                                         record=self.recorded,
                                                         clock=clock,
                                                         when='start',
                                                         timestep=int(round(self.sampling_interval/simulator.state.dt)))
        simulator.state.network.add(self._devices[variable])

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        self.sampling_interval = sampling_interval or self._simulator.state.dt
        if variable not in self._devices:
            self._create_device(self.population.brian_group, variable)
        #update StateMonitor.record and StateMonitor.recordindex
        if not variable is 'spikes':
            device = self._devices[variable]
            device.record = numpy.sort(numpy.fromiter(self.recorded[variable], dtype=int)) - self.population.first_id
            device.recordindex = dict((i,j) for i,j in zip(device.record,
                                                           range(len(device.record))))
            logger.debug("recording %s from %s" % (variable, self.recorded[variable]))

    def _reset(self):
        """Clear the list of cells to record."""
        for device in self._devices.values():
            device.reinit()
            device.record = False

    def _clear_simulator(self):
        """Delete all recorded data, but retain the list of cells to record from."""
        for device in self._devices.values():
            device.reinit()

    def _get_spiketimes(self, id):
        i = id - self.population.first_id
        return self._devices['spikes'].spiketimes[i]/ms

    def _get_all_signals(self, variable, ids, clear=False):
        # need to filter according to ids
        device = self._devices[variable]
        # because we use `when='start'`, need to add the value at the end of the final time step.
        values = numpy.array(device._values)
        #print(ids)
        #print(device.record)
        current_values = device.P.state_(device.varname)[device.record]
        all_values = numpy.vstack((values, current_values[numpy.newaxis, :]))
        logging.debug("@@@@ %s %s %s", id(device), values.shape, all_values.shape)
        varname = self.population.celltype.state_variable_translations[variable]['translated_name']
        all_values = eval(self.population.celltype.state_variable_translations[variable]['reverse_transform'], {}, {varname: all_values})
        if clear:
            self._devices[variable].reinit()
        return all_values

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding      = self.population.first_id
        indices = numpy.fromiter(filtered_ids, dtype=int) - padding
        for i, id in zip(indices, filtered_ids):
            N[id] = len(self._devices['spikes'].spiketimes[i])
        return N
