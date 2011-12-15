"""
Current source classes for the neuron module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

from neuron import h

def _iclamp_property(local_name, hoc_name):
    """
    Returns a new property for getting/setting iclamp attributes
    """
    def get(self):
        return getattr(self, local_name)
    def set(self, value):
        setattr(self, local_name, value)
        for iclamp in self._devices:
            setattr(iclamp, hoc_name, value)
    return property(fget=get, fset=set)


class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""
    pass


class DCSource(CurrentSource):
    """Source producing a single pulse of current of constant amplitude."""
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """Construct the current source.
        
        Arguments:
            start     -- onset time of pulse in ms
            stop      -- end of pulse in ms
            amplitude -- pulse amplitude in nA
        """
        self._amplitude = amplitude
        self._start = start
        self._stop = stop or 1e12
        self._devices = []
    
    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        for id in cell_list:
            if id.local:
                iclamp = h.IClamp(0.5, sec=id._cell.source_section)
                iclamp.delay = self._start
                iclamp.dur = self._stop-self._start
                iclamp.amp = self._amplitude
                self._devices.append(iclamp)

    amplitude = _iclamp_property("_amplitude", "amp")
    
    def _set_start(self, value):
        self._start = value
        for iclamp in self._devices:
            iclamp.delay = value
            iclamp.dur = self._stop - value
    start = property(fget=lambda self: self._start,
                     fset=_set_start)
    
    def _set_stop(self, value):
        self._stop = value
        for iclamp in self._devices:
            iclamp.dur = value - self._start
    stop = property(fget=lambda self: self._stop,
                    fset=_set_stop)


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
        self.times = h.Vector(times)
        self.amplitudes = h.Vector(amplitudes)
        self._devices = []
    
    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        for id in cell_list:
            if id.local:
                if not id.celltype.injectable:
                    raise TypeError("Can't inject current into a spike source.")
                iclamp = h.IClamp(0.5, sec=id._cell.source_section)
                iclamp.delay = 0.0
                iclamp.dur = 1e12
                iclamp.amp = 0.0
                self._devices.append(iclamp)
                self.amplitudes.play(iclamp._ref_amp, self.times)
                
    # need to make times and amplitudes properties so they can be modified
    