# encoding: utf-8

import numpy
from NeuroTools.parameters import ParameterSet
from NeuroTools.stgen import StGen
from simple_network import SimpleNetwork
from multisim import MultiSim
from pyNN import nest2, neuron, pcsim

trigger_spike = 77.1
sim_list = [nest2, neuron, pcsim]
#sim_list = [neuron, pcsim]
sim_time = 200.0
spike_interval = 20.0
stgen = StGen()
stgen.seed(834585)

parameters = ParameterSet({
    'system': { 'timestep': 0.1, 'min_delay': 0.1, 'max_delay': 10.0 },
    'input_spike_times': stgen.poisson_generator(1.0/spike_interval, sim_time), #numpy.arange(5, sim_time, 10.0),
    'trigger_spike_times': stgen.poisson_generator(1.0/spike_interval, sim_time),
    'cell_type': 'IF_curr_exp',
    'cell_parameters': { 'tau_refrac': 10.0, 'tau_m': 2.0, 'tau_syn_E': 1.0 },
    'plasticity': { 'short_term': None,
                    'long_term': {
                        'timing_dependence': { 'model': 'SpikePairRule',
                                               'params': { 'tau_plus': 20.0,
                                                           'tau_minus': 20.0 }},
                        'weight_dependence': { 'model': 'AdditiveWeightDependence',
                                               'params': { 'w_min': 0, 'w_max': 0.4,
                                                           'A_plus': 0.01, 'A_minus': 0.01 }},
                        'ddf': 1.0,
                    }     
                  },
    'weights': 0.01,
    'delays': 1.0,
})
    
networks = MultiSim(sim_list, SimpleNetwork, parameters)
    
networks.add_population("trigger", 1, 'SpikeSourceArray', {'spike_times': parameters.trigger_spike_times})
networks.add_projection("trigger", "post", "AllToAllConnector", {'weights': 100.0, 'delays': 0.1})

networks.run(0) # needed for PCSIM to set its weights properly
networks.save_weights()

networks.run(sim_time, int(sim_time), networks.save_weights)

spike_data = networks.get_spikes()
vm_data = networks.get_v()
w_data = networks.get_weights()

import pylab
pylab.rcParams['interactive'] = True

# plot Vm
pylab.figure(1)
for sim_name, vm in vm_data.items():
    pylab.plot(vm['post'][:,1], vm['post'][:,2], label="post (%s)" % sim_name)
pylab.legend(loc='upper left')

# plot weights
key = "pre→post"
pylab.figure(2)
for sim_name, w in w_data.items():
    pylab.plot(w[key][:,0], w[key][:,1], label="%s (%s)" % (key, sim_name))
pylab.legend(loc='upper right')

# plot spikes
pylab.figure(3)
for i, (sim_name, spikes) in enumerate(spike_data.items()):
    pylab.plot( spikes['post'][:,1], i*numpy.ones_like(spikes['post'][:,1]),
               "|", label="Postsynaptic spikes (%s)" % sim_name, markersize=100)
pylab.plot( parameters.input_spike_times, (i+1)*numpy.ones_like(parameters.input_spike_times),
           "|", label="Presynaptic spikes", markersize=100 )
pylab.ylim(-0.5,i+1.5)