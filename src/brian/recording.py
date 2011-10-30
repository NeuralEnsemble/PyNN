"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import brian
from pyNN import recording
from pyNN.brian import simulator
import logging

mV = brian.mV
ms = brian.ms
uS = brian.uS

logger = logging.getLogger("PyNN")

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
  
    def __init__(self, variable, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, variable, population, file)
        self._devices = [] # defer creation until first call of record()
    
    def _create_devices(self, group):
        """Create a Brian recording device."""
        clock = simulator.state.simclock
        if self.variable == 'spikes':
            devices = [brian.SpikeMonitor(group, record=self.recorded)]
        elif self.variable == 'v':
            devices = [brian.StateMonitor(group, 'v', record=True, clock=clock)]
        elif self.variable == 'gsyn':
            example_cell = list(self.recorded)[0]
            varname = example_cell.celltype.synapses['excitatory']
            device1 = brian.StateMonitor(group, varname, record=True, clock=clock)
            varname = example_cell.celltype.synapses['inhibitory']
            device2 = brian.StateMonitor(group, varname, record=True, clock=clock)
            devices = [device1, device2]
        else:
            devices = [brian.StateMonitor(group, self.variable, record=self.recorded, clock=clock)]
        for device in devices:
            simulator.state.add(device)
        return devices
    
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        #update StateMonitor.record and StateMonitor.recordindex
        self.recorded = self.recorded.union(ids)
        if len(self._devices) == 0:
            self._devices = self._create_devices(ids[0].parent_group)
        if not self.variable is 'spikes':
            cells              = list(self.recorded)
            for device in self._devices:
                device.record      = numpy.array(cells) - cells[0].parent.first_id
                device.recordindex = dict((i,j) for i,j in zip(device.record,
                                                            range(len(device.record))))
            logger.debug("recording %s from %s" % (self.variable, cells))
    
    def _reset(self):
        raise NotImplementedError("Recording reset is not currently supported for pyNN.brian")

    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        filtered_ids = self.filter_recorded(filter)
        cells        = list(filtered_ids) 
        padding      = cells[0].parent.first_id        
        filtered_ids = numpy.array(cells) - padding
        if self.variable == 'spikes':
            data    = numpy.empty((0,2))
            for id in filtered_ids:
                times    = self._devices[0].spiketimes[id]/ms
                new_data = numpy.array([numpy.ones(times.shape)*id + padding, times]).T
                data     = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            values = numpy.array(self._devices[0]._values)/mV
            times  = self._devices[0].times/ms
            data   = numpy.empty((0,3))
            for id, row in zip(self.recorded, values.T):
                new_data = numpy.array([numpy.ones(row.shape)*id, times, row]).T
                data = numpy.concatenate((data, new_data))
            if filter is not None:
                mask = reduce(numpy.add, (data[:,0]==id for id in filtered_ids + padding))
                data = data[mask]
        elif self.variable == 'gsyn':
            values1 = numpy.array(self._devices[0]._values)/uS
            values2 = numpy.array(self._devices[1]._values)/uS
            times   = self._devices[0].times/ms
            data    = numpy.empty((0,4))
            for id, row1, row2 in zip(self.recorded, values1.T, values2.T):
                assert row1.shape == row2.shape
                new_data = numpy.array([numpy.ones(row1.shape)*id, times, row1, row2]).T
                data = numpy.concatenate((data, new_data))
            if filter is not None:
                mask = reduce(numpy.add, (data[:,0]==id for id in filtered_ids + padding))
                data = data[mask]
        else:
            values = numpy.array(self._devices[0]._values)/mV
            times  = self._devices[0].times/ms
            data   = numpy.empty((0,3))
            for id, row in zip(self.recorded, values.T):
                new_data = numpy.array([numpy.ones(row.shape)*id, times, row]).T
                data = numpy.concatenate((data, new_data))
            if filter is not None:
                mask = reduce(numpy.add, (data[:,0]==id for id in filtered_ids + padding))
                data = data[mask]
        return data

    def _local_count(self, filter=None):
        N = {}
        filtered_ids = self.filter_recorded(filter)
        cells        = list(filtered_ids)
        padding      = cells[0].parent.first_id
        filtered_ids = numpy.array(cells) - padding   
        for id in filtered_ids:
            N[id + padding] = len(self._devices[0].spiketimes[id])
        return N
        

simulator.Recorder = Recorder
