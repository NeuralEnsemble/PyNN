#import pyNN.neuron as sim
from pyNN import common
import pyNN.random
import numpy
from numpy import exp, cos, pi
from NeuroTools import stgen

def set_simulator(sim):
    global BabblingPopulation
    BabblingPopulation = babbling_population_factory(sim)

class BabblingPopulation:
    def __init__(*args, **kwargs):
        raise Exception("You must first initialize the babble module using `set_simulator(sim)`")

def babbling_population_factory(sim):
    class BabblingPopulation(sim.Population):
        
        def __init__(self, dims, Rmax=60, Rmin=0, Rsigma=0.2, correlation_time = 20, rng=None, label=None):
            assert isinstance(dims, int) or len(dims)==1, "Only 1-D populations supported for now"
            sim.Population.__init__(self, dims, cellclass=sim.SpikeSourceArray, label=label)
            self.positions /= self.size # set positions to between 0 and 1
            self.rng = rng or pyNN.random.NumpyRNG()
            assert isinstance(self.rng, pyNN.random.NumpyRNG)
            self.correlation_time = correlation_time
            self.tuning_curve = lambda x, x0: (Rmax-Rmin)*exp((cos(2*pi*(x-x0))-1)/(Rsigma*Rsigma) ) + Rmin
            self.stgen = stgen.StGen(numpyrng=self.rng)
            self.t = 0 # real t
            self.tn = 0 # nominal t
        
        def generate_spikes(self, duration):
            actual_duration = duration + self.tn - self.t
            print "Actual duration = ", actual_duration
            position_changes = self.stgen.poisson_generator(1.0/self.correlation_time, actual_duration) # time relative to self.t
            position_changes = numpy.concatenate(([0.0], position_changes)) # add a change at time 0
            positions = self.rng.uniform(0, 1, len(position_changes))
            for cell in self:
                rates = 0.001 * self.tuning_curve(cell.position[0], positions) # spikes/ms
                # not sure this is the most efficient method, since the ratio between
                # the max rate and the min rate can be 1e20, so almost all the spikes
                # are thrown away during thinning
                #print rates
                #print position_changes
                cell.spike_times = self.t + self.stgen.poissondyn_generator(position_changes, rates, actual_duration)
                
            self.tn += duration
            self.t += position_changes[-1]
            
    return BabblingPopulation
    
if __name__ == "__main__":
    import pyNN.neuron as sim
    set_simulator(sim)
    sim.setup(use_cvode=True)
    sim.h.cvode.active(1)
    sim.h.cvode.use_local_dt(1)         # The variable time step method must be used.
    sim.h.cvode.condition_order(2)      # Improves threshold-detection.
    
    p = BabblingPopulation(10)
    print p.positions[0]
    p.record()
    p.generate_spikes(100)
    sim.run(100)
    p.generate_spikes(100)
    sim.run(100)
    print p.getSpikes()