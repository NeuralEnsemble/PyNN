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
        self._simulator.recorder_list.append(self)
        if self.variable is "spikes":
            self.data  = numpy.empty([0, 2])
        elif self.variable is "v":
            self.data  = numpy.empty([0, 3])
        elif self.variable is "gsyn":
            self.data  = numpy.empty([0, 4])
        else:
            raise Exception("Nemo can record only v and spikes for now !")    

    def write(self, file=None, gather=False, compatible_output=True, filter=None):
        recording.Recorder.write(self, file, gather, compatible_output, filter)
        #self._simulator.recorder_list.remove(self)

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        self.recorded = self.recorded.union(ids)
        
    def _reset(self):
        raise NotImplementedError("Recording reset is not currently supported for pyNN.nemo")

    def _add_spike(self, fired, time):
        ids       = self.recorded.intersection(fired)
        self.data = numpy.vstack((self.data, numpy.array([list(ids), [time]*len(ids)]).T)) 
        ## To file or memory ? ###

    def _add_vm(self, time):
        data      =  self._simulator.state.sim.get_membrane_potential(list(self.recorded))   
        self.data = numpy.vstack((self.data, numpy.array([list(self.recorded), [time]*len(self.recorded), data]).T))

    def _add_gsyn(self, time):
        ge      =  self._simulator.state.sim.get_neuron_state(list(self.recorded), 1)
        gi      =  self._simulator.state.sim.get_neuron_state(list(self.recorded), 2) 
        self.data = numpy.vstack((self.data, numpy.array([list(self.recorded), [time]*len(self.recorded), ge, gi]).T))

    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        filtered_ids = self.filter_recorded(filter)
        if len(self.data) > 0:
            mask = reduce(numpy.add, (self.data[:,0]==id for id in filtered_ids))                            
            data = self.data[mask]
            return data
        else:
            return self.data

    def _local_count(self, filter=None):
        N = {}
        filtered_ids = self.filter_recorded(filter)
        cells        = list(filtered_ids)
        filtered_ids = numpy.array(cells)   
        for id in filtered_ids:
            N[id] = 0
        spikes = self._get(gather=False, compatible_output=False, filter=filter)
        ids   = numpy.sort(spikes[:,0].astype(int))
        idx   = numpy.unique(ids)
        left  = numpy.searchsorted(ids, idx, 'left')
        right = numpy.searchsorted(ids, idx, 'right')
        for id, l, r in zip(idx, left, right):
            N[id] = r-l
        return N
        

simulator.Recorder = Recorder
