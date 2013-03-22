"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import recording
from pyNN.moose import simulator

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
        
    def _record(self, new_ids):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if self.variable == 'spikes':
            for id in new_ids:
                id._cell.record_spikes()
        elif self.variable == 'v':
            for id in new_ids:
                id._cell.record_v()
        elif self.variable == 'gsyn':
            for id in new_ids:
                id._cell.record_gsyn("excitatory")
                id._cell.record_gsyn("inhibitory")
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
    
    def _reset(self):
        raise NotImplementedError("TO DO")
    
    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        # compatible_output is not used, but is needed for compatibility with the nest module.
        # Does nest really need it?                
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.recorded:
                spikes = 1e3*numpy.array(id._cell.spike_table) # convert from s to ms
                spikes = spikes[spikes<=simulator.state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape)*id, spikes]).T
                    data = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.recorded:
                v = 1e3*numpy.array(id._cell.vmTable) # convert from V to mV
                t = simulator.state.dt*numpy.arange(0.0, v.size)
                new_data = numpy.array([numpy.ones(v.shape)*id, t, v]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            data = numpy.empty((0,4))
            for id in self.recorded:
                ge = 1e6*numpy.array(id._cell.gsyn_trace['excitatory']) # convert from S to uS
                gi = 1e6*numpy.array(id._cell.gsyn_trace['inhibitory'])
                t = simulator.state.dt*numpy.arange(0.0, ge.size)           
                new_data = numpy.array([numpy.ones(ge.shape)*id, t, ge, gi]).T
                data = numpy.concatenate((data, new_data))
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
    #    if gather and simulator.state.num_processes > 1:
    #        data = recording.gather(data)
    
        if filter is not None:
            filtered_ids = self.filter_recorded(filter)
            mask = reduce(numpy.add, (data[:,0]==id for id in filtered_ids))
            data = data[mask]
        return data
        
    def _local_count(self, filter=None):
        N = {}
        for id in self.filter_recorded(filter):
            N[int(id)] = len(id._cell.spike_table)
        return N

simulator.Recorder = Recorder
