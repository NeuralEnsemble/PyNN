"""
:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import pypcsim
from pyNN import recording
from pyNN.pcsim import simulator

# --- For implementation of record_X()/get_X()/print_X() -----------------------

class Recorder(recording.Recorder):
    """Encapsulates data and functions related to recording model variables."""
    _simulator = simulator
    
    fieldnames = {'v': 'Vm',
                  'gsyn': 'psr'}
    
    def __init__(self, variable, population=None, file=None):
        __doc__ = recording.Recorder.__doc__
        recording.Recorder.__init__(self, variable, population, file)
        self.recorders = {}
    
    def _record(self, new_ids):
        """Called by record()"""
        net = simulator.net
        if self.variable == 'spikes':        
            for id in new_ids:
                #if self.population:
                #    pcsim_id = self.population.pcsim_population[int(id)]
                #else:
                pcsim_id = int(id)
                src_id = pypcsim.SimObject.ID(pcsim_id)
                rec = net.create(pypcsim.SpikeTimeRecorder(),
                                 pypcsim.SimEngine.ID(src_id.node, src_id.eng))            
                net.connect(pcsim_id, rec, pypcsim.Time.sec(0))
                assert id not in self.recorders
                self.recorders[id] = rec
        elif self.variable == 'v':
            for id in new_ids:
                #if self.population:
                #    pcsim_id = self.population.pcsim_population[int(id)]
                #else:
                pcsim_id = int(id)
                src_id = pypcsim.SimObject.ID(pcsim_id)
                rec = net.create(pypcsim.AnalogRecorder(),
                                 pypcsim.SimEngine.ID(src_id.node, src_id.eng))
                net.connect(pcsim_id, Recorder.fieldnames[self.variable], rec, 0, pypcsim.Time.sec(0))
                self.recorders[id] = rec
        else:
            raise NotImplementedError("Recording of %s not implemented." % self.variable)

    def _reset(self):
        raise NotImplementedError("TO DO")

    def _get(self, gather=False, compatible_output=True, filter=None):
        """Return the recorded data as a Numpy array."""
        # compatible_output is not used, but is needed for compatibility with the nest module.
        # Does nest really need it?
        net = simulator.net
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.filter_recorded(filter):
                rec = self.recorders[id]
                if isinstance(net.object(id), pypcsim.SpikingInputNeuron):
                    spikes = 1000.0*numpy.array(net.object(id).getSpikeTimes()) # is this special case really necessary?
                    spikes = spikes[spikes<=simulator.state.t]
                else:
                    spikes = 1000.0*numpy.array(net.object(rec).getSpikeTimes())
                spikes = spikes.flatten()
                spikes = spikes[spikes<=simulator.state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape, dtype=int)*id, spikes]).T
                    data = numpy.concatenate((data, new_data))           
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.filter_recorded(filter):
                rec = self.recorders[id]
                v = 1000.0*numpy.array(net.object(rec).getRecordedValues())
                v = v.flatten()
                final_v = 1000.0*net.object(id).getVm()
                v = numpy.append(v, final_v)
                dt = simulator.state.dt
                t = dt*numpy.arange(v.size)
                new_data = numpy.array([numpy.ones(v.shape, dtype=int)*id, t, v]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            raise NotImplementedError
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
        if gather and simulator.state.num_processes > 1:
            data = recording.gather(data)
        return data

    def count(self, gather=False, filter=None):
        """
        Return the number of data points for each cell, as a dict. This is mainly
        useful for spike counts or for variable-time-step integration methods.
        """
        N = {}
        if self.variable == 'spikes':
            for id in self.filter_recorded(filter):
                N[id] = simulator.net.object(self.recorders[id]).spikeCount()
        else:
            raise Exception("Only implemented for spikes.")
        if gather and simulator.state.num_processes > 1:
            N = recording.gather_dict(N)
        return N
    
simulator.Recorder = Recorder
