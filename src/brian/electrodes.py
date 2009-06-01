"""
Current source classes for the brian module.

Classes:
    DCSource           -- a single pulse of current of constant amplitude.
    StepCurrentSource  -- a step-wise time-varying current.

$Id:$
"""

from brian import ms, nA, network_operation
from simulator import state, net
import numpy

current_sources = []

@network_operation(when='start')
def update_currents():
    global current_sources
    for current_source in current_sources:
        current_source._update_current()
net.add(update_currents)


class CurrentSource(object):
    """Base class for a source of current to be injected into a neuron."""

    def __init__(self):
        global current_sources
        self.cell_list = []
        current_sources.append(self)

    def inject_into(self, cell_list):
        """Inject this current source into some cells."""
        self.cell_list.extend(cell_list)        
    
        
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
        CurrentSource.__init__(self)
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        self.times = times
        self.amplitudes = amplitudes
        self.i = 0
        self.running = True
    
    def _update_current(self):
        """
        Check whether the current amplitude needs updating, and then do so if
        needed.
        
        This is called at every timestep.
        """
        if self.running and state.t >= self.times[self.i]: #*ms:
            amp = self.amplitudes[self.i]*nA               
            self.i += 1
            if self.i >= len(self.times):
                self.running = False
            #print self.i, state.t, amp
            for cell in self.cell_list:
                cell.parent_group[int(cell)].i_inj = amp
        
                
class DCSource(StepCurrentSource):
    """Source producing a single pulse of current of constant amplitude."""
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        """Construct the current source.
        
        Arguments:
            start     -- onset time of pulse in ms
            stop      -- end of pulse in ms
            amplitude -- pulse amplitude in nA
        """
        times = [0.0, start, (stop or 1e99)]
        amplitudes = [0.0, amplitude, 0.0]
        StepCurrentSource.__init__(self, times, amplitudes)
        