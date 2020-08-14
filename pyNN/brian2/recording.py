"""

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import numpy
import quantities as pq
import brian2
from pyNN.core import is_listlike
from pyNN import recording
from . import simulator

mV = brian2.mV
ms = brian2.ms
uS = brian2.uS
pq.uS = pq.UnitQuantity('microsiemens', 1e-6 * pq.S, 'uS')
pq.nS = pq.UnitQuantity('nanosiemens', 1e-9 * pq.S, 'nS')

logger = logging.getLogger("PyNN")


class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator

    def __init__(self, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, population, file)
        self._devices = {}  # defer creation until first call of run()

    def _create_device(self, group, variable):
        """Create a Brian2 recording device."""
        # Brian2 records in the 'start' scheduling slot by default
        clock = simulator.state.network.clock
        if variable == 'spikes':
            self._devices[variable] = brian2.SpikeMonitor(group, record=self.recorded) #TODO: Brian2 SpikeMonitor check exists
        else:
            varname = self.population.celltype.state_variable_translations[variable]['translated_name']
            neurons_to_record = numpy.sort(numpy.fromiter(self.recorded[variable], dtype=int)) - self.population.first_id
            self._devices[variable] = brian2.StateMonitor(group, varname,
                                                          record=neurons_to_record,
                                                          clock=clock,
                                                          when='start')#,
                                                          #dt=int(round(self.sampling_interval / simulator.state.dt))*brian2.ms)
        simulator.state.network.add(self._devices[variable])

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        self.sampling_interval = sampling_interval or self._simulator.state.dt

    def _finalize(self):
        for variable in self.recorded:
            if variable not in self._devices:
                self._create_device(self.population.brian2_group, variable)
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
        if is_listlike(id):
            all_spiketimes = {}
            for cell_id in id:
                i = cell_id - self.population.first_id
                spiky=self._devices['spikes'].spike_trains()
                all_spiketimes[cell_id] = spiky[i] / ms
            return all_spiketimes
        else:
            i = id - self.population.first_id
            spiky=self._devices['spikes'].spike_trains()
            return spiky[i] / ms

    def _get_all_signals(self, variable, ids, clear=False):
        # need to filter according to ids
        device = self._devices[variable]
        # because we use `when='start'`, need to add the value at the end of the final time step.
        device.record_single_timestep()
        #values = numpy.array(device._values)
        varname = self.population.celltype.state_variable_translations[variable]['translated_name']
        values = getattr(device, varname)[0]  # [0] ####### LOOOOK HERE
        values = self.population.celltype.state_variable_translations[variable]['reverse_transform'](values)
        if clear:
            self._devices[variable].reinit()
        return values

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding = self.population.first_id
        indices = numpy.fromiter(filtered_ids, dtype=int) - padding
        spiky=self._devices['spikes'].spike_trains()
        for i, id in zip(indices, filtered_ids):
            #N[id] = len(self._devices['spikes'].spiketimes[i])
            N[id] = len(spiky[i])
        return N
