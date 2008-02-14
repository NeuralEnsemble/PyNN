"""
Provides a Population of spike sources with firing rate correlations.

We suppose there is a point stimulus whose position jumps about rapidly in space.
The mean firing rate of each neuron at a given point in time depends on the
position of the stimulus within its receptive field (bell-shaped tuning curve).
For a given mean firing rate, each neuron generates spikes according to a
Poisson process (i.e., overall, each neuron generates an inhomogenous Poisson
process with step changes in mean rate).

The names `babble` and `BabblingPopulation` come from this random jumping about,
which is analogous to verbal babbling and to the apparently random limb
movements made by newborn infants ('motor babbling').

Several populations can 'observe' the same stimulus - this is implemented by
a `sync` argument which gives the population whose `stim_positions` and
`position_changes` attributes should be used.

Arbitrary transformations can be applied to the stimulus positions, e.g.
consider observing a stimulus through prism glasses.

The current version has periodic boundary conditions.

For a reference, see:
  Davison A.P. and Frégnac Y. (2006) Learning crossmodal spatial transformations
  through spike-timing-dependent plasticity. J. Neurosci 26: 5604-5615.

Based on an original NEURON model, see:
  http://senselab.med.yale.edu/senselab/ModelDB/ShowModel.asp?model=64261

$Id$
"""

from pyNN import common
import pyNN.random
import numpy
from numpy import exp, cos, pi
from NeuroTools import stgen


def set_simulator(sim):
    global BabblingPopulation
    BabblingPopulation = babbling_population_factory(sim)


class BabblingPopulation:
    """
    Population of spike sources generating inhomogenous Poisson processes
    with step changes in mean rate. Mean rate is correlated between neurons.
    
    This is a non-functional stub. To obtain a functional class for your
    simulator, initialize the babble module using `set_simulator(sim)`.
    """
    def __init__(*args, **kwargs):
        raise Exception("You must first initialize the babble module using `set_simulator(sim)`")


def babbling_population_factory(sim):
    """Return a BabblingPopulation class for a specific stimulator."""
    
    class BabblingPopulation(sim.Population):
        """
        Population of spike sources generating inhomogenous Poisson processes
        with step changes in mean rate. Mean rate is correlated between
        neurons.
        """
        
        def __init__(self, dims, Rmax=60.0, Rmin=0.0, Rsigma=0.2, alpha=1.0,
                     correlation_time=20.0, transform=None, rng=None, label=None):
            """
            `Rmax`, `Rmin` and `Rsigma` are the parameters of the bell-shaped
            tuning curve. The amplitude of the entire curve may be scaled by
            `alpha`.
            
            `correlation_time` is the mean interval (exponential distribution)
            between stimulus position changes.
            
            `transform` is a function taking a single float (or 1D array of
            floats) as an argument and returning the same.
            
            `rng` may be any of the random number generators from `pyNN.random`.
            If it is not specified, a new NumpyRNG is created.
            """
            assert isinstance(dims, int) or len(dims)==1, "Only 1-D populations supported for now"
            sim.Population.__init__(self, dims, cellclass=sim.SpikeSourceArray, label=label)
            self.positions /= self.size # set positions to between 0 and 1
            self.rng = rng or pyNN.random.NumpyRNG()
            assert isinstance(self.rng, pyNN.random.NumpyRNG)
            self.correlation_time = correlation_time
            self.transform = transform
            self.tuning_curve = lambda x, x0: alpha*( (Rmax-Rmin)*exp( (cos(2*pi*(x-x0))-1)/(Rsigma*Rsigma) ) + Rmin)
            self.stgen = stgen.StGen(numpyrng=self.rng)
            self.t = 0 # real t
            self.tn = 0 # nominal t
        
        def generate_spikes(self, duration, sync=None):
            """
            Generate spikes for all neurons in the population, starting from the
            end of the last spike sequence (or zero if this is the first).
            
            Since the times for which the stimulus stays in one location are
            random, the sequence may end slightly before `duration`, so the
            actual end time is returned.
            
            If `sync` is not set or is set to `self`, the stimulus position
            changes are generated here. If it is set to another
            `BabblingPopulation` object, the stimulus positions are taken from
            that object. Make sure that `generate_spikes()` is called for the
            `sync` object first!
            """
            actual_duration = duration + self.tn - self.t # if the previous sequence ended early, we need to start from that time, `t`, not the nominal time `tn`
            sync = sync or self
            if sync == self:
                sync.position_changes = self.stgen.poisson_generator(1.0/self.correlation_time, actual_duration) # time relative to self.t
                sync.position_changes = numpy.concatenate(([0.0], sync.position_changes)) # add a change at time 0
                sync.stim_positions = self.rng.uniform(0, 1, len(sync.position_changes))
            if self.transform:
                stim_positions = self.transform(sync.stim_positions)
            else:
                stim_positions = sync.stim_positions
            for cell in self:
                rates = 0.001 * self.tuning_curve(cell.position[0], stim_positions) # spikes/ms
                # not sure this is the most efficient method, since the ratio between
                # the max rate and the min rate can be 1e20, so almost all the spikes
                # are thrown away during thinning
                cell.spike_times = self.t + self.stgen.poissondyn_generator(sync.position_changes,
                                                                            rates,
                                                                            actual_duration)
                
            self.tn += duration
            self.t += sync.position_changes[-1]
            return self.t
            
    return BabblingPopulation
    
    
if __name__ == "__main__": # simple test
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