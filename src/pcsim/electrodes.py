
import numpy
import pypcsim
from pyNN.pcsim import simulator


class CurrentSource(object):


    def inject_into(self, cell_list):
        for cell in cell_list:
            c = simulator.net.connect(self.input_node, cell, pypcsim.StaticAnalogSynapse())
            self.connections.append(c)
    
        
class StepCurrentSource(CurrentSource):
    
    def __init__(self, times, amplitudes):
        CurrentSource.__init__(self)
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        self.times = times
        self.amplitudes = amplitudes
        n = len(times)
        durations = numpy.empty((n+1,))
        levels = numpy.empty_like(durations)
        durations[0] = times[0]
        levels[0] = 0.0
        t = numpy.array(times)
        try:
            durations[1:-1] = t[1:] - t[0:-1]
        except ValueError, e:
            raise ValueError("%s. durations[1:].shape=%s, t[1:].shape=%s, t[0:-1].shape=%s" % (e, durations[1:].shape, t[1:].shape, t[0:-1].shape))
        levels[1:] = amplitudes[:]
        durations[-1] = 1e12
        levels *= 1e-9    # nA --> A
        durations *= 1e-3 # s --> ms
        self.input_node = simulator.net.create(pypcsim.AnalogLevelBasedInputNeuron(levels, durations))
        self.connections = []
        
class DCSource(StepCurrentSource):
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        times = [0.0, start, (stop or 1e99)]
        amplitudes = [0.0, amplitude, 0.0]
        StepCurrentSource.__init__(self, times, amplitudes)
        