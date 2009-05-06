
from brian import ms, nA, network_operation
from simulator import state, net
import numpy

current_sources = []

@network_operation(when='start')
def update_currents():
    global current_sources
    for current_source in current_sources:
        current_source.update_current()
net.add(update_currents)


class CurrentSource(object):

    def __init__(self):
        global current_sources
        self.cell_list = []
        current_sources.append(self)

    def inject_into(self, cell_list):
        self.cell_list.extend(cell_list)        
    
        
class StepCurrentSource(CurrentSource):
    
    def __init__(self, times, amplitudes):
        CurrentSource.__init__(self)
        assert len(times) == len(amplitudes), "times and amplitudes must be the same size (len(times)=%d, len(amplitudes)=%d" % (len(times), len(amplitudes))
        self.times = times
        self.amplitudes = amplitudes
        self.i = 0
        self.running = True
    
    def update_current(self):    
        if self.running and state.t >= self.times[self.i]: #*ms:
            amp = self.amplitudes[self.i]*nA               
            self.i += 1
            if self.i >= len(self.times):
                self.running = False
            #print self.i, state.t, amp
            for cell in self.cell_list:
                cell.parent_group[int(cell)].i_inj = amp
        
                
class DCSource(StepCurrentSource):
    
    def __init__(self, amplitude=1.0, start=0.0, stop=None):
        times = [0.0, start, (stop or 1e99)]
        amplitudes = [0.0, amplitude, 0.0]
        StepCurrentSource.__init__(self, times, amplitudes)
        