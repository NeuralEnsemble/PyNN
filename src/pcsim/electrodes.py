

class CurrentSource(object):


    def inject_into(self, cell_list):
        raise NotImplementedError
    
        
class StepCurrentSource(CurrentSource):
    
    def __init__(self, times, amplitudes):
        CurrentSource.__init__(self)
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        self.times = times
        self.amplitudes = amplitudes
        #AnalogLevelBasedInputNeuron(levels, durations)
        
        
class DCSource(StepCurrentSource):
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        times = [0.0, start, (stop or 1e99)]
        amplitudes = [0.0, amplitude, 0.0]
        StepCurrentSource.__init__(self, times, amplitudes)
        