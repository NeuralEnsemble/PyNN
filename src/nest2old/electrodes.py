import nest
import numpy

# should really use the StandardModel machinery to allow reverse translations

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
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        nest.SetStatus(self._device, {'amplitude_times': numpy.array(times, 'float'),
                                      'amplitude_values': 1000.0*numpy.array(amplitudes, 'float')})
        
class NoisyCurrentSource(CurrentSource):
    
    def __init__(self, mean, stdev, start=0.0, stop=None, frozen=False):
        self._device = nest.Create('noise_generator')
        nest.SetStatus(self._device, {'mean': mean*1000.0,
                                      'std': stdev*1000.0,
                                      'start': float(start),
                                      'frozen': frozen})
        if stop:
            nest.SetStatus(self._device, {'stop': float(stop)})