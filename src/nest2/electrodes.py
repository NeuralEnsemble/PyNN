import nest
import numpy

class CurrentSource(object):
    
    def inject_into(self, cell_list):
        nest.Connect(self._device, cell_list)
    

class DCSource(CurrentSource):
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """amplitude is in nA"""
        self.amplitude = amplitude
        self._device = nest.Create('dc_generator')
        nest.SetStatus(self._device, {'amplitude': 1000.0*self.amplitude,
                                      'start': float(start)}) # conversion from nA to pA
        if stop:
            nest.SetStatus(self._device, {'stop': float(stop)})
        
    
    
class StepCurrentSource(CurrentSource):
    
    def __init__(self, times, amplitudes):
        self._device = nest.Create('step_current_generator')
        nest.SetStatus(self._device, {'amplitude_times': numpy.array(times, 'float'),
                                      'amplitude_values': 1000.0*numpy.array(amplitudes, 'float')})
        