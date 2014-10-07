"""
Simple network with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_exp neurons.

Andrew Davison, UNIC, CNRS
August 2006, November 2009

"""

import socket, os
from importlib import import_module
import numpy
from pyNN.utility import get_script_args, init_logging, normalized_filename

simulator_name = get_script_args(1)[0]
sim = import_module("pyNN.%s" % simulator_name)

from pyNN.random import NumpyRNG, RandomDistribution

init_logging(None, debug=True)

seed = 764756387
rng = NumpyRNG(seed=seed, parallel_safe=True)
tstop = 1000.0 # ms
input_rate = 100.0 # Hz
cell_params = {'tau_refrac': 2.0,  # ms
               'v_thresh':  -50.0, # mV
               'tau_syn_E':  2.0,  # ms
               'tau_syn_I':  2.0,  # ms
               'tau_m': RandomDistribution('uniform', low=18.0, high=22.0, rng=rng)
}
n_record = 3

node = sim.setup(timestep=0.025, min_delay=1.0, max_delay=1.0, debug=True, quit_on_end=False)
print("Process with rank %d running on %s" % (node, socket.gethostname()))

print("[%d] Creating populations" % node)
n_spikes = int(2*tstop*input_rate/1000.0)
spike_times = numpy.add.accumulate(rng.next(n_spikes, 'exponential',
                                            {'beta': 1000.0/input_rate}, mask_local=False))

input_population  = sim.Population(10, sim.SpikeSourceArray(spike_times=spike_times), label="input")
output_population = sim.Population(20, sim.IF_curr_exp(**cell_params), label="output")
print("[%d] input_population cells: %s" % (node, input_population.local_cells))
print("[%d] output_population cells: %s" % (node, output_population.local_cells))
print("tau_m =", output_population.get('tau_m'))

print("[%d] Connecting populations" % node)
connector = sim.FixedProbabilityConnector(0.5, rng=rng)
syn = sim.StaticSynapse(weight=1.0)
projection = sim.Projection(input_population, output_population, connector, syn)

filename = normalized_filename("Results", "simpleRandomNetwork", "conn",
                               simulator_name, sim.num_processes())
projection.save('connections', filename)

input_population.record('spikes')
output_population.record('spikes')
output_population.sample(n_record, rng).record('v')

print("[%d] Running simulation" % node)
sim.run(tstop)

print("[%d] Writing spikes and Vm to disk" % node)
filename = normalized_filename("Results", "simpleRandomNetwork_output", "pkl",
                               simulator_name, sim.num_processes())
output_population.write_data(filename, annotations={'script_name': __file__})
##input_population.write_data('%s_input.h5' % file_stem)
spike_counts = output_population.get_spike_counts()
for id in sorted(spike_counts):
    print(id, spike_counts[id])

print("[%d] Finishing" % node)
sim.end()
print("[%d] Done" % node)
