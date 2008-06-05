"""
Simple network with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_alpha neurons.

Andrew Davison, UNIC, CNRS
August 2006

$Id:$
"""

import sys
from NeuroTools.stgen import StGen

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python

exec("from pyNN.%s import *" % simulator)

tstop = 1000.0

setup(timestep=0.025,min_delay=1.0,max_delay=1.0,file="simpleNetwork2.xml")
    
cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
output_population = Population(2, IF_cond_exp, cell_params, "output")

spikeGenerator = StGen()
spike_times = list(spikeGenerator.poisson_generator(100.0/1000.0,tstop)) # rate in spikes/ms

input_population  = Population(1, SpikeSourceArray, {'spike_times': spike_times }, "input")

projection = Projection(input_population, output_population, 'allToAll')
projection.setWeights(1.0)

input_population.record()
output_population.record()
output_population.record_v()

run(tstop)

output_population.printSpikes("simpleNetwork_output_%s.ras" % simulator)
input_population.printSpikes("simpleNetwork_input_%s.ras" % simulator)
output_population.print_v("simpleNetwork_output_%s.v" % simulator)

end()

