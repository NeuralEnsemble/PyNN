"""
A demonstration of the use of callbacks to update the spike times in a SpikeSourceArray.

Usage: update_spike_source_array.py [-h] [--plot-figure] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
"""

import numpy as np
from pyNN.utility import get_simulator, normalized_filename, ProgressBar
from pyNN.utility.plotting import Figure, Panel
from pyNN.parameters import Sequence

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.",
                              {"action": "store_true"}))

rate_increment = 20
interval = 200


class SetRate(object):
    """
    A callback which changes the firing rate of a population of spike
    sources at a fixed interval.
    """

    def __init__(self, population, rate_generator, update_interval=20.0):
        assert isinstance(population.celltype, sim.SpikeSourceArray)
        self.population = population
        self.update_interval = update_interval
        self.rate_generator = rate_generator

    def __call__(self, t):
        try:
            rate = next(rate_generator)
            if rate > 0:
                isi = 1000.0/rate
                times = t + np.arange(0, self.update_interval, isi)
                # here each neuron fires with the same isi,
                # but there is a phase offset between neurons
                spike_times = [
                    Sequence(times + phase * isi)
                    for phase in self.population.annotations["phase"]
                ]
            else:
                spike_times = []
            self.population.set(spike_times=spike_times)
        except StopIteration:
            pass
        return t + self.update_interval


class MyProgressBar(object):
    """
    A callback which draws a progress bar in the terminal.
    """

    def __init__(self, interval, t_stop):
        self.interval = interval
        self.t_stop = t_stop
        self.pb = ProgressBar(width=int(t_stop / interval), char=".")

    def __call__(self, t):
        self.pb(t / self.t_stop)
        return t + self.interval


sim.setup()


# === Create a population of poisson processes ===============================

p = sim.Population(50, sim.SpikeSourceArray())
p.annotate(phase=np.random.uniform(0, 1, size=p.size))
p.record('spikes')


# === Run the simulation, with two callback functions ========================

rate_generator = iter(range(0, 100, rate_increment))
sim.run(1000, callbacks=[MyProgressBar(10.0, 1000.0),
                         SetRate(p, rate_generator, interval)])


# === Retrieve recorded data, and count the spikes in each interval ==========

data = p.get_data().segments[0]

all_spikes = np.hstack([st.magnitude for st in data.spiketrains])
spike_counts = [((all_spikes >= x) & (all_spikes < x + interval)).sum()
                for x in range(0, 1000, interval)]
expected_spike_counts = [p.size * rate * interval / 1000.0
                         for rate in range(0, 100, rate_increment)]

print("\nActual spike counts: {}".format(spike_counts))
print("Expected mean spike counts: {}".format(expected_spike_counts))

if options.plot_figure:
    Figure(
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True, markersize=0.5),
        title="Incrementally updated SpikeSourceArrays",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(normalized_filename("Results", "update_spike_source_array", "png", options.simulator))

sim.end()
