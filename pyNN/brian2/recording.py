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
        self._devices = {}  # defer creation until first call of record()

    def _create_device(self, group, variable):
        """Create a Brian2 recording device."""
        # Brian2 records in the 'start' scheduling slot by default
        clock = simulator.state.network.clock
        if variable == 'spikes':
            self._devices[variable] = brian2.SpikeMonitor(group, record=self.recorded) #TODO: Brian2 SpikeMonitor check exists
        else:
            varname = self.population.celltype.state_variable_translations[variable]['translated_name']
            self._devices[variable] = brian2.StateMonitor(group, varname,
                                                         record=list(dict(self.recorded)[varname]),
                                                         clock=clock,
                                                         when='start')#,
                                                         #dt=int(round(self.sampling_interval / simulator.state.dt))*brian2.ms)
        simulator.state.network.add(self._devices[variable])

    def _record(self, variable, new_ids, sampling_interval=None):
        """Add the cells in `new_ids` to the set of recorded cells."""
        self.sampling_interval = sampling_interval or self._simulator.state.dt
        if variable not in self._devices:
            self._create_device(self.population.brian2_group, variable)
        # update StateMonitor.record and StateMonitor.record
        if variable is not 'spikes':
            device = self._devices[variable]
            device.record = numpy.sort(numpy.fromiter(self.recorded[variable], dtype=int)) - self.population.first_id
            device.record = dict((i, j) for i, j in zip(device.record,
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
        if is_listlike(id):
            all_spiketimes = {}
            for cell_id in id:
                i = cell_id - self.population.first_id
                all_spiketimes[cell_id] = self._devices['spikes'].spiketimes[i] / ms
            return all_spiketimes
        else:
            i = id - self.population.first_id
            return self._devices['spikes'].spiketimes[i] / ms

    def _get_all_signals(self, variable, ids, clear=False):
        # need to filter according to ids
        device = self._devices[variable]
        # because we use `when='start'`, need to add the value at the end of the final time step.
        print "device = ", device
        # values = numpy.array(device._values)
        print "variable = ", variable
        print "ids = ", ids
        values = eval('device.{}'.format(variable))[0]
        vm = values
        print "A >> len(vm) = ", len(vm), ", min = ", min(vm), ", max = ", max(vm)
        # print "values = ", values
        print "len(values) = ", len(values)
        # print "device.varname = ", device.varname
        print "device.record = ", device.record
        print "device.variables = ", device.variables
        print "device.record_variables = ", device.record_variables
        print "device.variables['v'] = ", device.variables['v']
        print "device.variables['v'].name = ", device.variables['v'].name
        print "device.variables['v'].dim = ", device.variables['v'].dim
        # print "device.variables['v'].get_value() = ", device.variables['v'].get_value()
        vm = device.variables['v'].get_value()
        print "B >> len(vm) = ", len(vm), ", min = ", min(vm), ", max = ", max(vm)
        print "len(device.variables['v'].get_value()) = ", len(device.variables['v'].get_value())
        # current_values = device.P.state_(device.varname)[device.record]
        # all_values = numpy.vstack((values, current_values[numpy.newaxis, :]))
        # logging.debug("@@@@ %s %s %s", id(device), values.shape, all_values.shape)
        # varname = self.population.celltype.state_variable_translations[variable]['translated_name']
        # all_values = eval(self.population.celltype.state_variable_translations[variable]['reverse_transform'], {}, {varname: all_values})
        if clear:
            self._devices[variable].reinit()
        return values
        # return all_values

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding = self.population.first_id
        indices = numpy.fromiter(filtered_ids, dtype=int) - padding
        for i, id in zip(indices, filtered_ids):
            N[id] = len(self._devices['spikes'].spiketimes[i])
        return N
