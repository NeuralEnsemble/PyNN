"""
For each standard cell type available, creates a simple network with a single
poisson spike sources projecting to a single standard standard cell, runs the
model with each simulator, and checks that the difference in the membrane
potential trace of the post-synaptic cell between simulators is within some
tolerance.
"""

from time import time
import numpy
from NeuroTools.parameters import ParameterSet
from NeuroTools.stgen import StGen
from pyNN.utility import MultiSim
from pyNN import nest2old, neuron, pcsim
from simple_network import SimpleNetwork
from pprint import pprint

sim_list = [nest2old, neuron, pcsim]
sim_time = 1000.0
spike_interval = 1.0

stgen = StGen()
seed = int(1e9*(time()%1))
stgen.seed(seed)

parameters = ParameterSet({
    'system': { 'timestep': 0.01, 'min_delay': 0.1, 'max_delay': 10.0 },
    'input_spike_times': stgen.poisson_generator(1000.0/spike_interval, t_stop=sim_time, array=True),
    'cell_type': 'IF_curr_exp',
    'cell_parameters': { 'tau_refrac': 2.0, 'tau_m': 20.0, 'tau_syn_E': 2.0 },
    'plasticity': { 'short_term': None, 'long_term': None },
    'weights': 0.5,
    'delays': 1.0,
})

networks = MultiSim(sim_list, SimpleNetwork, parameters)
networks.run(sim_time)

spike_data = networks.get_spikes()
vm_data = networks.get_v()

def calc_distances():
    distances = {'victorpurpura': {}, 'kreuz': {}}
    for measure in distances.keys():
        for sim1 in sim_list:
            distances[measure][sim1.__name__] = {} 
            for sim2 in sim_list:
                f_distance = getattr(spike_data[sim1.__name__]['post'][0], "distance_%s" % measure)
                distances[measure][sim1.__name__][sim2.__name__] = f_distance(spike_data[sim2.__name__]['post'][0])
    return distances
   

def plot_figures():
    import pylab
    pylab.rcParams['interactive'] = True
    
    # plot Vm
    pylab.figure(1)
    for sim_name, vm in vm_data.items():
        vm['post'].plot(display=pylab.gca(), kwargs={'label': "post (%s)" % sim_name})
    pylab.legend(loc='upper left')
    ##pylab.ylim(-80, -40)
    
    # plot spikes
    pylab.figure(2)
    for i, (sim_name, spikes) in enumerate(spike_data.items()):
        if len(spikes['post']) > 0:
            pylab.plot( spikes['post'][0].spike_times, (2*i)*numpy.ones_like(spikes['post'][0].spike_times),
                       "|", label="Postsynaptic spikes (%s)" % sim_name, markersize=50)
        if len(spikes['pre']) > 0:
            print sim_name, len(spikes['pre'])
            pylab.plot( spikes['pre'][0].spike_times, (2*i+1)*numpy.ones_like(spikes['pre'][0].spike_times),
                       "|", label="Presynaptic spikes (%s)" % sim_name, markersize=50)
    pylab.plot( parameters.input_spike_times, (2*i+2)*numpy.ones_like(parameters.input_spike_times),
               "|", label="Presynaptic spikes", markersize=50 )
    pylab.ylim(-0.5,2*i+2.5)

print spike_data
print vm_data
distances = calc_distances()
pprint(distances)    
#plot_figures()