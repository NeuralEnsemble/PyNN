import numpy
import brian
from pyNN import recording
from pyNN.brian import simulator

mV = brian.mV
ms = brian.ms
uS = brian.uS

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
  
    def __init__(self, variable, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, variable, population, file)
        self._devices = [] # defer creation until first call of record()
    
    def _create_devices(self, group):
        """Create a Brian recording device."""
        clock = simulator.state.simclock
        if self.variable == 'spikes':
            devices = [brian.SpikeMonitor(group, record=True)]
        elif self.variable == 'v':
            devices = [brian.StateMonitor(group, 'v', record=True, clock=clock)]
        elif self.variable == 'gsyn':
            example_cell = list(self.recorded)[0]
            varname = example_cell.cellclass.synapses['excitatory']
            device1 = brian.StateMonitor(group, varname, record=True, clock=clock)
            varname = example_cell.cellclass.synapses['inhibitory']
            device2 = brian.StateMonitor(group, varname, record=True, clock=clock)
            devices = [device1, device2]
        for device in devices:
            simulator.net.add(device)
        return devices
    
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        #update StateMonitor.record and StateMonitor.recordindex
        self.recorded = self.recorded.union(ids)
        if len(self._devices) == 0:
            self._devices = self._create_devices(ids[0].parent_group)
        if self.variable is not 'spikes':
            for device in self._devices:
                device.record = list(self.recorded)
                device.recordindex = dict((i,j) for i,j in zip(device.record,
                                                               range(len(device.record))))
    
    def _get(self, gather=False, compatible_output=True):
        """Return the recorded data as a Numpy array."""
        if self.variable == 'spikes':
            data = numpy.array([(id, time/ms) for (id, time) in self._devices[0].spikes if id in self.recorded])
        elif self.variable == 'v':
            values = self._devices[0].values/mV
            times = self._devices[0].times/ms
            data = numpy.empty((0,3))
            for id, row in enumerate(values):
                new_data = numpy.array([numpy.ones(row.shape)*id, times, row]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            values1 = self._devices[0].values/uS
            values2 = self._devices[1].values/uS
            times = self._devices[0].times/ms
            data = numpy.empty((0,4))
            for id, (row1, row2) in enumerate(zip(values1, values2)):
                assert row1.shape == row2.shape
                new_data = numpy.array([numpy.ones(row1.shape)*id, times, row1, row2]).T
                data = numpy.concatenate((data, new_data))
        return data

simulator.Recorder = Recorder