"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import recording
from pyNN.neuron import simulator
import re
from neuron import h

recordable_pattern = re.compile(r'((?P<section>\w+)(\((?P<location>[-+]?[0-9]*\.?[0-9]+)\))?\.)?(?P<var>\w+)')

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
    
    def _record(self, new_ids):
        """Add the cells in `new_ids` to the set of recorded cells."""
        if self.variable == 'spikes':
            for id in new_ids:
                id._cell.record(1)
        elif self.variable == 'v':
            for id in new_ids:
                id._cell.record_v(1)
        elif self.variable == 'gsyn':
            for id in new_ids:
                id._cell.record_gsyn("excitatory", 1)
                id._cell.record_gsyn("inhibitory", 1)
                if id._cell.excitatory_TM is not None:
                    id._cell.record_gsyn("excitatory_TM", 1)
                    id._cell.record_gsyn("inhibitory_TM", 1)
        else:
            for id in new_ids:
               self._native_record(id)
    
    def _reset(self):
        for id in self.recorded:
            id._cell.traces = {}
            id._cell.record(active=False)
            id._cell.record_v(active=False)
            for syn_name in id._cell.gsyn_trace:
                id._cell.record_gsyn(syn_name, active=False)
    
    def _native_record(self, id):
        match = recordable_pattern.match(self.variable)
        if match:
            parts = match.groupdict()
            if parts['section']:
                section = getattr(id._cell, parts['section'])
                if parts['location']:
                    segment = section(float(parts['location']))
                else:
                    segment = section
            else:
                segment = id._cell.source
            id._cell.traces[self.variable] = vec = h.Vector()
            vec.record(getattr(segment, "_ref_%s" % parts['var']))
            if not id._cell.recording_time:
                id._cell.record_times = h.Vector()
                id._cell.record_times.record(h._ref_t)
                id._cell.recording_time += 1
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
    
    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        # compatible_output is not used, but is needed for compatibility with the nest module.
        # Does nest really need it?
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.filter_recorded(filter):
                spikes = numpy.array(id._cell.spike_times)
                spikes = spikes[spikes<=simulator.state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape)*id, spikes]).T
                    data = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.filter_recorded(filter):
                v = numpy.array(id._cell.vtrace)  
                t = numpy.array(id._cell.record_times)               
                new_data = numpy.array([numpy.ones(v.shape)*id, t, v]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            data = numpy.empty((0,4))
            for id in self.filter_recorded(filter):
                ge = numpy.array(id._cell.gsyn_trace['excitatory'])
                gi = numpy.array(id._cell.gsyn_trace['inhibitory'])
                if 'excitatory_TM' in id._cell.gsyn_trace:
                    ge_TM = numpy.array(id._cell.gsyn_trace['excitatory_TM'])
                    gi_TM = numpy.array(id._cell.gsyn_trace['inhibitory_TM'])
                    if ge.size == 0:
                        ge = ge_TM
                    elif ge.size == ge_TM.size:
                        ge = ge + ge_TM
                    else:
                        raise Exception("Inconsistent conductance array sizes: ge.size=%d, ge_TM.size=%d", (ge.size, ge_TM.size))
                    if gi.size == 0:
                        gi = gi_TM
                    elif gi.size == gi_TM.size:
                        gi = gi + gi_TM
                    else:
                        raise Exception()
                t = numpy.array(id._cell.record_times)             
                new_data = numpy.array([numpy.ones(ge.shape)*id, t, ge, gi]).T
                data = numpy.concatenate((data, new_data))
        else:
            data = numpy.empty((0,3))
            for id in self.filter_recorded(filter):
                var = numpy.array(id._cell.traces[self.variable])  
                t = numpy.array(id._cell.record_times)               
                new_data = numpy.array([numpy.ones(var.shape)*id, t, var]).T
                data = numpy.concatenate((data, new_data))    
            #raise Exception("Recording of %s not implemented." % self.variable)
        if gather and simulator.state.num_processes > 1:
            data = recording.gather(data)
        return data
        
    def _local_count(self, filter=None):
        N = {}
        for id in self.filter_recorded(filter):
            N[int(id)] = id._cell.spike_times.size()
        return N

simulator.Recorder = Recorder
