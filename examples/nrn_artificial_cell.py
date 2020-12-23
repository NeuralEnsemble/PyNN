# encoding: utf-8
"""
Small network created with NEURON ARTIFICIAL_CELL models


Usage: nrn_artificial_cell.py

"""

import numpy as np
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.parameters import Sequence
from pyNN.random import RandomDistribution as rnd
import pyNN.neuron as sim


# === Define parameters ========================================================

n = 20      # Number of cells
w = 0.1  # synaptic weight (dimensionless)
cell_params = {
    'tau'    : 20.0,   # (ms)
    'refrac' : 2.0,    # (ms)
}
dt         = 0.1           # (ms)
syn_delay  = 1.0           # (ms)
input_rate = 50.0          # (Hz)
simtime    = 1000.0        # (ms)

# === Build the network ========================================================

sim.setup(timestep=dt, max_delay=syn_delay)

cells = sim.Population(n, sim.IntFire1(**cell_params),
                       initial_values={'m': rnd('uniform', (0.0, 1.0))},
                       label="cells")

number = int(2 * simtime * input_rate / 1000.0)
np.random.seed(26278342)


def generate_spike_times(i):
    gen = lambda: Sequence(np.add.accumulate(np.random.exponential(1000.0 / input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()
assert generate_spike_times(0).max() > simtime

spike_source = sim.Population(n, sim.SpikeSourceArray(spike_times=generate_spike_times))

spike_source.record('spikes')
cells.record('spikes')
cells[0:2].record('m')

syn = sim.StaticSynapse(weight=w, delay=syn_delay)
input_conns = sim.Projection(spike_source, cells, sim.FixedProbabilityConnector(0.5), syn,
                             receptor_type="default")

# === Run simulation ===========================================================

sim.run(simtime)

filename = normalized_filename("Results", "nrn_artificial_cell", "pkl",
                               "neuron", sim.num_processes())
cells.write_data(filename, annotations={'script_name': __file__})

print("Mean firing rate: ", cells.mean_spike_count() * 1000.0 / simtime, "Hz")

plot_figure = True
if plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    data = cells.get_data().segments[0]
    m = data.filter(name="m")[0]
    Figure(
        Panel(m, ylabel="Membrane potential (dimensionless)", yticks=True, ylim=(0, 1)),
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True),
        annotations="Simulated with NEURON"
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
