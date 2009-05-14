"""
Simple network with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_alpha neurons.

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python

exec("from pyNN.%s import *" % simulator)
from pyNN.random import NumpyRNG

from NeuroTools.stgen import StGen

tstop = 1000.0

myid = setup(timestep=0.025,min_delay=1.0,max_delay=1.0)

cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
output_population = Population(20, IF_curr_alpha, cell_params, "output")

spikeGenerator = StGen()
spike_times = list(spikeGenerator.poisson_generator(100.0/1000.0,tstop)) # rate in spikes/ms

input_population  = Population(3, SpikeSourceArray, {'spike_times': spike_times }, "input")

connector = FixedProbabilityConnector(0.5, weights=1.0)
rng = NumpyRNG(seed=764756387, parallel_safe=True, rank=myid, num_processes=num_processes())
projection = Projection(input_population, output_population, connector, rng=rng)

file_stem = "Results/simpleRandomNetwork_%s_np%d" % (simulator, num_processes())
projection.saveConnections('%s.conn' % file_stem)

input_population.record()
output_population.record()
output_population.record_v()

run(tstop)

output_population.printSpikes('%s_output.ras.%d' % (file_stem, myid))
input_population.printSpikes('%s_input.ras.%d' % (file_stem, myid))
output_population.print_v('%s_output.v.%d' % (file_stem, myid))

end()

