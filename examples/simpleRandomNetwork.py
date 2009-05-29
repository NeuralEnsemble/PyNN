"""
Simple network with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_alpha neurons.

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(__file__, 1)[0]  
exec("from pyNN.%s import *" % simulator_name)

from pyNN.random import NumpyRNG

seed = 764756387
tstop = 1000.0 # ms
input_rate = 100.0 # Hz
cell_params = {'tau_refrac': 2.0,  # ms
               'v_thresh':  -50.0, # mV
               'tau_syn_E':  2.0,  # ms
               'tau_syn_I':  2.0}  # ms


setup(timestep=0.025, min_delay=1.0, max_delay=1.0)

rng = NumpyRNG(seed=seed, parallel_safe=True, rank=rank(), num_processes=num_processes())

n_spikes = int(2*tstop*input_rate/1000.0)
spike_times = numpy.add.accumulate(rng.next(n_spikes, 'exponential', [1000.0/input_rate]))
input_population  = Population(3, SpikeSourceArray, {'spike_times': spike_times }, "input")

output_population = Population(20, IF_curr_alpha, cell_params, "output")

connector = FixedProbabilityConnector(0.5, weights=1.0)
projection = Projection(input_population, output_population, connector, rng=rng)

file_stem = "Results/simpleRandomNetwork_np%d_%s" % (num_processes(), simulator_name)
projection.saveConnections('%s.conn' % file_stem)

input_population.record()
output_population.record()
output_population.record_v()

run(tstop)

output_population.printSpikes('%s_output.ras' % file_stem)
input_population.printSpikes('%s_input.ras' % file_stem)
output_population.print_v('%s.v' % file_stem)

end()

