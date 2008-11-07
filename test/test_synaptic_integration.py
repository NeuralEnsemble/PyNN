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
from pyNN import nest2, neuron, pcsim
from simple_network import SimpleNetwork

sim_list = [nest2, neuron, pcsim]
sim_time = 1000.0
spike_interval = 1.0

stgen = StGen()
seed = int(1e9*(time()%1))
stgen.seed(seed)

parameters = ParameterSet({
    'system': { 'timestep': 0.01, 'min_delay': 0.1, 'max_delay': 10.0 },
    'input_spike_times': stgen.poisson_generator(1.0/spike_interval, sim_time),
    'cell_type': 'IF_curr_exp',
    'cell_parameters': { 'tau_refrac': 2.0, 'tau_m': 20.0, 'tau_syn_E': 2.0 },
    'plasticity': { 'short_term': None, 'long_term': None },
    'weights': 0.1,
    'delays': 1.0,
})

networks = MultiSim(sim_list, SimpleNetwork, parameters)
networks.run(sim_time)

spike_data = networks.get_spikes()
vm_data = networks.get_v()

print spike_data

def plot_figures():
    import pylab
    pylab.rcParams['interactive'] = True
    
    # plot Vm
    pylab.figure(1)
    for sim_name, vm in vm_data.items():
        pylab.plot(vm['post'][:,1], vm['post'][:,2], label="post (%s)" % sim_name)
    pylab.legend(loc='upper left')
    ##pylab.ylim(-80, -40)
    
    
    # plot spikes
    pylab.figure(2)
    for i, (sim_name, spikes) in enumerate(spike_data.items()):
        if len(spikes['post']) > 0:
            pylab.plot( spikes['post'][:,1], (2*i)*numpy.ones_like(spikes['post'][:,1]),
                       "|", label="Postsynaptic spikes (%s)" % sim_name, markersize=50)
        if len(spikes['pre']) > 0:
            print sim_name, len(spikes['pre'])
            pylab.plot( spikes['pre'][:,1], (2*i+1)*numpy.ones_like(spikes['pre'][:,1]),
                       "|", label="Presynaptic spikes (%s)" % sim_name, markersize=50)
    pylab.plot( parameters.input_spike_times, (2*i+2)*numpy.ones_like(parameters.input_spike_times),
               "|", label="Presynaptic spikes", markersize=50 )
    pylab.ylim(-0.5,2*i+2.5)
    
plot_figures()