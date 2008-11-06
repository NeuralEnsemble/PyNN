# encoding: utf-8

import numpy
from NeuroTools.parameters import ParameterSet
from simple_network import SimpleNetwork
from multisim import MultiSim
from pyNN import nest2, neuron, pcsim

trigger_spike = 80

parameters = ParameterSet({
    'system': { 'timestep': 0.1, 'min_delay': 0.1, 'max_delay': 10.0 },
    'input_spike_times': numpy.arange(5, trigger_spike+10.0, 10.0),
    'cell_type': 'IF_curr_exp',
    'cell_parameters': { 'tau_refrac': 10.0 },
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
    'weights': 0.1,
    'delays': 1.0,
})
    
networks = MultiSim((nest2, neuron, pcsim), SimpleNetwork, parameters)
    
networks.add_population("trigger", 1, 'SpikeSourceArray', {'spike_times': [trigger_spike]})
networks.add_projection("trigger", "post", "AllToAllConnector", {'weights': 10.0, 'delays': 0.1})
networks.save_weights()

networks.run(100, 100, "save_weights")

#spikes = net.get_spikes()
#vm = net.get_v()
#w = net.get_weights()
#
#import pylab
#pylab.rcParams['interactive'] = True
#
## plot Vm
#pylab.figure(1)
#pylab.plot(vm['post'][:,1], vm['post'][:,2], label='post')
#pylab.legend(loc='upper right')
#
## plot weights
#key = "pre→post"
#pylab.figure(2)
#pylab.plot(w[key][:,0], w[key][:,1], label=key)
#pylab.legend(loc='upper right')
