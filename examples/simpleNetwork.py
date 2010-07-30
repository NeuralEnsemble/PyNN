"""
Simple network with a Poisson spike source projecting to a pair of IF_curr_alpha neurons

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

import numpy
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)

tstop = 1000.0
rate = 100.0

setup(timestep=0.1, min_delay=0.2, max_delay=1.0)
    
cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
output_population = Population(2, IF_curr_alpha, cell_params, label="output")

number = int(2*tstop*rate/1000.0)
numpy.random.seed(26278342)
spike_times = numpy.add.accumulate(numpy.random.exponential(1000.0/rate, size=number))
assert spike_times.max() > tstop
print spike_times.min()

input_population  = Population(1, SpikeSourceArray, {'spike_times': spike_times }, label="input")

projection = Projection(input_population, output_population, AllToAllConnector())
projection.setWeights(1.0)

input_population.record()
output_population.record()
output_population.record_v()

run(tstop)

output_population.printSpikes("Results/simpleNetwork_output_%s.ras" % simulator_name)
input_population.printSpikes("Results/simpleNetwork_input_%s.ras" % simulator_name)
output_population.print_v("Results/simpleNetwork_%s.v" % simulator_name)

end()

