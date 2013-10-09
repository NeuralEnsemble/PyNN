"""
A demonstration of the use of callbacks to vary the rate of a SpikeSourcePoisson.

Every 200 ms, the Poisson firing rate is increased by 20 spikes/s
"""

from pyNN.utility import get_simulator, normalized_filename, ProgressBar
from pyNN.utility.plotting import Figure, Panel

sim, args = get_simulator()


class SetRate(object):

    def __init__(self, population, rate_generator, interval=20.0):
        assert isinstance(population.celltype, sim.SpikeSourcePoisson)
        self.population = population
        self.interval = interval
        self.rate_generator = rate_generator
        
    def __call__(self, t):
        try:
            self.population.set(rate=next(rate_generator))
        except StopIteration:
            pass
        return t + self.interval


class MyProgressBar(object):
    
    def __init__(self, interval, t_stop):
        self.interval = interval
        self.t_stop = t_stop
        self.pb = ProgressBar(width=int(t_stop/interval), char=".")
        
    def __call__(self, t):
        self.pb(t/self.t_stop)
        return t + self.interval

    
sim.setup()

p = sim.Population(50, sim.SpikeSourcePoisson())
p.record('spikes')

rate_generator = iter(range(0, 100, 20))
progress_bar = ProgressBar()
sim.run(1000, callbacks=[SetRate(p, rate_generator, 200.0),
                         MyProgressBar(10.0, 1000.0)])

data = p.get_data().segments[0]

Figure(
    Panel(data.spiketrains, xlabel="Time (ms)", xticks=True),
    title="Time varying Poisson spike trains",
).save(normalized_filename("Results", "varying_poisson", "png", args.simulator))

sim.end()