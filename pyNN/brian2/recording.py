"""
Brian 2 implementation of recording machinery.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import numpy as np
import quantities as pq
import brian2
from .. import recording
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
        recording.Recorder.__init__(self, population, file)
        self._devices = {}  # defer creation until first call of run()

    def _create_device(self, group, variable):
        """Create a Brian2 recording device."""
        # Brian2 records in the 'start' scheduling slot by default
        if variable == 'spikes':
            self._devices[variable] = brian2.SpikeMonitor(group, record=self.recorded)
        else:
            translations = self.population.celltype.state_variable_translations
            varname = translations[variable]['translated_name']
            neurons_to_record = np.sort(np.fromiter(
                self.recorded[variable], dtype=int)) - self.population.first_id
            self._devices[variable] = brian2.StateMonitor(group, varname,
                                                          record=neurons_to_record,
                                                          when='end',
                                                          dt=self.sampling_interval * ms)
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
        self._devices = {}
        for device in self._devices.values():
            del device

    def _clear_simulator(self):
        """Delete all recorded data, but retain the list of cells to record from."""
        # for variable, device in self._devices.items():
        #     group = device.source
        #     self._create_device(group, variable)
        #     del device
        for device in self._devices.values():
            device.resize(0)

    def _get_spiketimes(self, requested_ids, clear=False):
        id_array = self._devices["spikes"].i + self.population.first_id
        times_array = self._devices["spikes"].t / ms
        mask = np.in1d(id_array, requested_ids)
        return id_array[mask], times_array[mask]

    def _get_all_signals(self, variable, ids, clear=False):
        # need to filter according to ids

        # check that the requested ids have indeed been recorded
        if not set(ids).issubset(self.recorded[variable]):
            raise Exception("You are requesting data from neurons that have not been recorded")
        device = self._devices[variable]
        varname = self.population.celltype.state_variable_translations[variable]['translated_name']
        if len(ids) == len(self.recorded[variable]):
            values = getattr(device, varname).T
        else:
            raise NotImplementedError  # todo - construct a mask to get only the desired signals
        translations = self.population.celltype.state_variable_translations[variable]
        values = translations['reverse_transform'](**{translations['translated_name']: values})
        # because we use `when='end'`, need to add the value at the beginning of the run
        tmp = np.empty((values.shape[0] + 1, values.shape[1]))
        tmp[1:, :] = values
        population_mask = self.population.id_to_index(ids)
        tmp[0, :] = self.population.initial_values[variable][population_mask]
        values = tmp
        if clear:
            self._devices[variable].resize(0)
        times = None
        return values, times

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding = self.population.first_id
        indices = np.fromiter(filtered_ids, dtype=int) - padding
        spiky = self._devices['spikes'].spike_trains()
        for i, id in zip(indices, filtered_ids):
            N[id] = len(spiky[i])
        return N
