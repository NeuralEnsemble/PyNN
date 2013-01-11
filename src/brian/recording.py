"""

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import brian
from pyNN import recording
from pyNN.brian import simulator
import logging
import neo
import quantities as pq

mV = brian.mV
ms = brian.ms
uS = brian.uS
pq.uS = pq.UnitQuantity('microsiemens', 1e-6*pq.S, 'uS')

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
                                                         record=self.recorded,
                                                         clock=clock,
                                                         when='start')
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

    @staticmethod
    def find_units(variable):
        return recording.UNITS_MAP.get(variable, "dimensionless")

    def _get_spiketimes(self, id):
        i = id - self.population.first_id
        return self._devices['spikes'].spiketimes[i]/ms

    def _get_all_signals(self, variable, ids):
        # need to filter according to ids
        device = self._devices[variable]
        # because we use `when='start'`, need to add the value at the end of the final time step.
        values = numpy.array(device._values)
        print ids
        print device.record
        current_values = device.P.state_(device.varname)[device.record]
        return numpy.vstack((values, current_values[numpy.newaxis, :]))

    def _local_count(self, variable, filter_ids=None):
        N = {}
        filtered_ids = self.filter_recorded(variable, filter_ids)
        padding      = self.population.first_id
        filtered_ids = numpy.fromiter(filtered_ids, dtype=int) - padding
        for id in filtered_ids:
            N[id] = len(self._devices['spikes'].spiketimes[id])
        return N
