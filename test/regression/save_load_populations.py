"""
Save and load populations.

Jens Kremkow, INCM, CNRS
November 2008
"""
import sys
simulator_name = sys.argv[-1]
exec("from pyNN.%s import *" % simulator_name)

setup()
# create population
dim = (2,2,1)
orientation = 1.
# neurons
neurons = Population(dim,IF_cond_exp,label='v1')
neurons.record()
neurons.orientation = orientation
# input
poisson_input = Population(dim,SpikeSourcePoisson,label='poisson')
poisson_input.set({'rate':1000.})
# Projection
s = Projection(poisson_input,neurons,OneToOneConnector())
s.setWeights(0.004)


# second neuron population
neurons2 = Population(dim,IF_cond_exp,label='v2')
neurons2.record_v()
# Projection
s = Projection(neurons,neurons2,OneToOneConnector())
s.setWeights(0.004)

run(500.)
v1 = neurons2.get_v()
# save population
filename = 'Results/save_load_populations.shelve'
save_population(neurons,filename,variables=['orientation'])


# load population
setup()
neurons_loaded = load_population(filename)

# second neuron population
neurons2 = Population(dim,IF_cond_exp,label='v2')
neurons2.record_v()
# Projection
s = Projection(neurons_loaded,neurons2,OneToOneConnector())
s.setWeights(0.004)

run(500.)
v2 = neurons2.get_v() 

neurons = numpy.unique(v1[:,0])
for neuron in neurons:
    a = v1[v1[:,0]==neuron] == v2[v2[:,0]==neuron]
    assert a.sum()/a.size == 1

assert neurons_loaded.orientation == orientation

end()
