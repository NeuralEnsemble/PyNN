"""
Save and load populations.

Jens Kremkow, INCM, CNRS
November 2008
"""
import sys
simulator_name = sys.argv[-1]
exec("import pyNN.%s as sim" % simulator_name)
from pyNN.utility import save_population, load_population
import numpy

sim.setup()
# create population
dim = (2,2,1)
orientation = 1.
# neurons
neurons = sim.Population(dim, sim.IF_cond_exp, label='v1')
neurons.record()
neurons.orientation = orientation
# input
poisson_input = sim.Population(dim, sim.SpikeSourcePoisson, label='poisson')
poisson_input.set({'rate':1000.})
# Projection
s = sim.Projection(poisson_input, neurons, sim.OneToOneConnector())
s.setWeights(0.004)


# second neuron population
neurons2 = sim.Population(dim, sim.IF_cond_exp, label='v2')
neurons2.record_v()
# Projection
s = sim.Projection(neurons, neurons2, sim.OneToOneConnector())
s.setWeights(0.004)

sim.run(500.)
v1 = neurons2.get_v()
# save population
filename = 'save_load_populations.shelve'
save_population(neurons, filename, variables=['orientation'])

# load population
sim.setup()
neurons_loaded = load_population(filename, sim)

# second neuron population
neurons2 = sim.Population(dim, sim.IF_cond_exp, label='v2')
neurons2.record_v()
# Projection
s = sim.Projection(neurons_loaded, neurons2, sim.OneToOneConnector())
s.setWeights(0.004)

sim.run(500.)
v2 = neurons2.get_v() 

neurons = numpy.unique(v1[:,0])
for neuron in neurons:
    a = v1[v1[:,0]==neuron] == v2[v2[:,0]==neuron]
    assert a.sum()/a.size == 1

assert neurons_loaded.orientation == orientation

sim.end()
