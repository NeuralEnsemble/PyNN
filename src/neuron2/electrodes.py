import neuron

class CurrentSource(object):
    pass

class DCSource(CurrentSource):
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """amplitude is in nA"""
        self.amplitude = amplitude
        self.start = start
        self.stop = stop or 1e12
        self._devices = []
    
    def inject_into(self, cell_list):
        for id in cell_list:
            self._devices.append(neuron.IClamp(id._cell,
                                               delay=self.start,
                                               dur=self.stop-self.start,
                                               amp=self.amplitude))
    
#class StepCurrentSource(CurrentSource):
#    
#    def __init__(self, times, amplitudes):
#        self.times = neuron.Vector(times)
#        self.amplitudes = neuron.Vector(amplitudes)
#        self._devices = []
#    
#    def inject_into(self, cell_list):
#        for id in cell_list:
#            iclamp = neuron.IClamp(id._cell, delay=0.0, dur=1e12, amp=0.0)
#            self._devices.append(iclamp)
#            self.amplitudes.hoc_obj.play(iclamp.hoc_obj._ref_amp, self.times)