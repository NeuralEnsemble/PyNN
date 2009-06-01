"""
Current source classes for the nest2 module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.
    NoisyCurrentSource -- a Gaussian whitish noise current.

$Id:$
"""

import nest
import numpy
from simulator import state
from pyNN.random import NumpyRNG, NativeRNG

# should really use the StandardModel machinery to allow reverse translations

class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""
    
    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        nest.DivergentConnect(self._device, cell_list)
    

class DCSource(CurrentSource):
    """Source producing a single pulse of current of constant amplitude."""
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """Construct the current source.
        
        Arguments:
            start     -- onset time of pulse in ms
            stop      -- end of pulse in ms
            amplitude -- pulse amplitude in nA
        """
        self.amplitude = amplitude
        self._device = nest.Create('dc_generator')
        nest.SetStatus(self._device, {'amplitude': 1000.0*self.amplitude,
                                      'start': float(start)}) # conversion from nA to pA
        if stop:
            nest.SetStatus(self._device, {'stop': float(stop)})
        
        
class StepCurrentSource(CurrentSource):
    """A step-wise time-varying current source."""
    
    def __init__(self, times, amplitudes):
        """Construct the current source.
        
        Arguments:
            times      -- list/array of times at which the injected current changes.
            amplitudes -- list/array of current amplitudes to be injected at the
                          times specified in `times`.
                          
        The injected current will be zero up until the first time in `times`. The
        current will continue at the final value in `amplitudes` until the end
        of the simulation.
        """
        self._device = nest.Create('step_current_generator')
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        times.append(1e12)                 # work around for 
        amplitudes.append(amplitudes[-1])  # bug in NEST
        nest.SetStatus(self._device, {'amplitude_times': numpy.array(times, 'float'),
                                      'amplitude_values': 1000.0*numpy.array(amplitudes, 'float')})
        
        
class NoisyCurrentSource(CurrentSource):
    """A Gaussian "white" noise current source. The current amplitude changes at fixed
    intervals, with the new value drawn from a Gaussian distribution."""
    # We have a possible problem here in that each recipient receives a
    # different noise stream, which is probably what is wanted in most
    # scenarios, but conflicts with the idea of a current source as a single
    # object.
    # For the purposes of reproducibility, it would also be nice to have
    # simulator-independent noise, which means adding an rng argument. If this
    # is a NativeRNG, we use the 'noise_generator' model, otherwise we generate
    # values and use a 'step_current_generator'.
    
    def __init__(self, mean, stdev, dt=None, start=0.0, stop=None, rng=None):
        """Construct the current source.
        
        Required arguments:
            mean  -- mean current amplitude in nA
            stdev -- standard deviation of the current amplitude in nA
            
        Optional arguments:
            dt    -- interval between updates of the current amplitude. Must be
                     a multiple of the simulation time step. If not specified,
                     the simulation time step will be used.
            start -- onset of the current injection in ms. If not specified, the
                     current will begin at the start of the simulation.
            stop  -- end of the current injection in ms. If not specified, the
                     current will continue until the end of the simulation.
            rng   -- an RNG object from the `pyNN.random` module. For speed,
                     this should be a `NativeRNG` instance (uses the simulator's
                     internal random number generator). For reproducibility
                     across simulators, use one of the other RNG types. If not
                     specified, a NumpyRNG is used.
        """
        self.rng = rng or NumpyRNG
        self.dt = dt or state.dt
        assert self.dt%dt == 0
        self.start = start
        self.stop = stop
        self.mean = mean
        self.stdev = stdev
        if isinstance(rng, NativeRNG):
            self._device = nest.Create('noise_generator')
            nest.SetStatus(self._device, {'mean': mean*1000.0,
                                          'std': stdev*1000.0,
                                          'start': float(start),
                                          'dt': dt})
            if stop:
                nest.SetStatus(self._device, {'stop': float(stop)})
        else:
            raise NotImplementedError("Only using a NativeRNG is currently supported.")