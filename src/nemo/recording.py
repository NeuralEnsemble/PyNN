"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import recording
from pyNN.nemo import simulator
import logging

logger = logging.getLogger("PyNN")

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
  
    def __init__(self, variable, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, variable, population, file)
        simulator.recorder_list.append(self)
        self.data  = {}    
        self.times = []    
    
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        self.recorded = self.recorded.union(ids)
        for id in self.recorded:
            self.data[id] = []
        
    def _reset(self):
        raise NotImplementedError("Recording reset is not currently supported for pyNN.nemo")

    def _add_spike(self, fired, time):
         idx   = list(self.recorded)
         if len(fired > 0):
             left  = numpy.searchsorted(fired, idx, 'left')
             right = numpy.searchsorted(fired, idx, 'right')
             for id, l, r in zip(idx, left, right):
                if l != r:
                    self.data[id] += [time]
        ## To file or memory ? ###

    def _add_vm(self, time):
        for id in list(self.recorded):
            self.data[id] += [simulator.state.sim.get_membrane_potential(int(id))]
        self.times += [time]

    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        filtered_ids = self.filter_recorded(filter)
        if self.variable == 'spikes':
            data    = numpy.empty((0,2))
            for id in filtered_ids:
                times    = numpy.array(self.data[id])
                new_data = numpy.array([numpy.ones(times.shape)*id, times]).T
                data     = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            data   = numpy.empty((0,3))
            N      = len(self.times)
            for id in filtered_ids:
                vm       = self.data[id]
                new_data = numpy.array([numpy.ones(N)*id, self.times, vm]).T
                data = numpy.concatenate((data, new_data))                    
        return data

    def _local_count(self, filter=None):
        N = {}
        filtered_ids = self.filter_recorded(filter)
        cells        = list(filtered_ids)
        filtered_ids = numpy.array(cells)   
        for id in filtered_ids:
            N[id] = len(self.data[id])
        return N
        

simulator.Recorder = Recorder
