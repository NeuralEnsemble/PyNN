"""
Simple network with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_exp neurons.

Andrew Davison, UNIC, CNRS
August 2006, November 2009

"""

import socket, os
import csa
import numpy
from pyNN.utility import get_script_args, Timer

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

from pyNN.random import NumpyRNG

timer = Timer()
seed = 764756387
tstop = 1000.0 # ms
input_rate = 100.0 # Hz
cell_params = {'tau_refrac': 2.0,  # ms
               'v_thresh':  -50.0, # mV
               'tau_syn_E':  2.0,  # ms
               'tau_syn_I':  2.0}  # ms
n_record = 5

node = setup(timestep=0.025, min_delay=1.0, max_delay=10.0, debug=True, quit_on_end=False)
print("Process with rank %d running on %s" % (node, socket.gethostname()))


rng = NumpyRNG(seed=seed, parallel_safe=True)

print("[%d] Creating populations" % node)
n_spikes = int(2*tstop*input_rate/1000.0)
spike_times = numpy.add.accumulate(rng.next(n_spikes, 'exponential',
                                            {'beta': 1000.0/input_rate}, mask_local=False))

input_population  = Population(100, SpikeSourceArray(spike_times=spike_times), label="input")
output_population = Population(10, IF_curr_exp(**cell_params), label="output")
print("[%d] input_population cells: %s" % (node, input_population.local_cells))
print("[%d] output_population cells: %s" % (node, output_population.local_cells))

print("[%d] Connecting populations" % node)
timer.start()
connector = CSAConnector(csa.random(0.5))
syn = StaticSynapse(weight=0.1)
projection = Projection(input_population, output_population, connector, syn)
print(connector.describe(), timer.elapsedTime())

file_stem = "Results/simpleRandomNetwork_csa_np%d_%s" % (num_processes(), simulator_name)

projection.save('all', '%s.conn' % file_stem)

input_population.record('spikes')
output_population.record('spikes')
output_population.sample(n_record, rng).record('v')

print("[%d] Running simulation" % node)
run(tstop)

print("[%d] Writing spikes and Vm to disk" % node)
output_population.write_data('%s_output.pkl' % file_stem)
#input_population.write_data('%s_input.pkl' % file_stem)

print("[%d] Finishing" % node)
end()
print("[%d] Done" % node)
