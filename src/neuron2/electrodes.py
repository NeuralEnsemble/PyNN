from neuron import h

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
            iclamp = h.IClamp(0.5, sec=id._cell)
            iclamp.delay = self.start
            iclamp.dur = self.stop-self.start
            iclamp.amp = self.amplitude
            self._devices.append(iclamp)
    
class StepCurrentSource(CurrentSource):
    
    def __init__(self, times, amplitudes):
        self.times = h.Vector(times)
        self.amplitudes = h.Vector(amplitudes)
        self._devices = []
    
    def inject_into(self, cell_list):
        for id in cell_list:
            iclamp = h.IClamp(0.5, sec=id._cell)
            iclamp.delay = 0.0
            iclamp.dur = 1e12
            iclamp.amp = 0.0
            self._devices.append(iclamp)
            self.amplitudes.play(iclamp._ref_amp, self.times)