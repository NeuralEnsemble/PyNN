"""
Simple network with a Poisson spike source projecting to a pair of
IF_curr_alpha neurons.  Based on simpleNetwork.py.

Mikael Djurfeldt, PDC, KTH
March 2012

"""

import numpy
from pyNN import music
from pyNN.utility import get_script_args

sim1,sim2 = music.setup(music.Config("nest", 1), music.Config("nest", 1))

tstop = 1000.0
rate = 100.0

test=music.multisim.ProxySimulator()
test.setup(timestep=0.1)
sim1.setup(timestep=0.1, min_delay=0.2, max_delay=1.0)
sim2.setup(timestep=0.1, min_delay=0.2, max_delay=1.0)

cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn_E':2.0, 'tau_syn_I':2.0}
output_population = sim2.Population(2, sim2.IF_curr_alpha, cell_params, label="output")

number = int(2*tstop*rate/1000.0)
numpy.random.seed(26278342)
spike_times = numpy.add.accumulate(numpy.random.exponential(1000.0/rate, size=number))

input_population  = sim1.Population(1, sim1.SpikeSourceArray, {'spike_times': spike_times}, label="input")

projection = music.Projection(input_population, output_population, sim2.AllToAllConnector())
projection.set('weight', 1.0)

input_population.record()
output_population.record()
output_population.record_v()

music.run(tstop)

output_population.printSpikes("Results/simpleNetwork_output_%s.ras" % 'nest')
input_population.printSpikes("Results/simpleNetwork_input_%s.ras" % 'nest')
output_population.print_v("Results/simpleNetwork_%s.v" % 'nest')

music.end()
