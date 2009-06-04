# encoding: utf-8
"""
For each simulator:
  - creates a simple network with two independent Poisson spike sources
    connected to a post-synaptic cell. One of the spike sources ('input') is
    connected with an STDP synapse with a very small weight. The other
    ('trigger') is connected with a static synapse and a very large weight, such
    that it should always cause the post-synaptic cell to fire.
  - records the evolution of the synaptic weight.
Calculates the difference in weight trajectories between the simulators

(Consider rewriting to conform to the same structure as in test_synaptic_integration.py)
"""

import numpy
from time import time
from pyNN import nest, neuron, pcsim
from pyNN.utility import MultiSim
from NeuroTools.parameters import ParameterSet
from NeuroTools.stgen import StGen
from simple_network import SimpleNetwork
from calc import STDPSynapse

PLOT_FIGURES = True
sim_list = [nest, neuron, pcsim]
sim_time = 200.0
spike_interval = 20.0 # ms
recording_interval = 1.0
stgen = StGen()
seed = int(1e9*(time()%1))
stgen.seed(seed)

parameters = ParameterSet({
    'system': { 'timestep': 0.01, 'min_delay': 0.1, 'max_delay': 10.0 },
    'input_spike_times': stgen.poisson_generator(rate=1000.0/spike_interval, t_stop=sim_time, array=True),
    'trigger_spike_times': stgen.poisson_generator(rate=1000.0/spike_interval, t_stop=sim_time, array=True),
    'cell_type': 'IF_curr_exp',
    'cell_parameters': { 'tau_refrac': 10.0, 'tau_m': 2.0, 'tau_syn_E': 1.0 },
    'plasticity': { 'short_term': None,
                    'long_term': {
                        'timing_dependence': { 'model': 'SpikePairRule',
                                               'params': { 'tau_plus': 20.0,
                                                           'tau_minus': 20.0 }},
                        'weight_dependence': { 'model': 'AdditiveWeightDependence',
                                               'params': { 'w_min': 0, 'w_max': 0.1,
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

networks.run(sim_time, int(sim_time/recording_interval), networks.save_weights)

spike_data = networks.get_spikes()
vm_data = networks.get_v()
w_data = networks.get_weights(at_input_spiketimes=True, recording_interval=recording_interval)
w_data1 = networks.get_weights(at_input_spiketimes=False)

P = parameters.plasticity.long_term
stdp_params = {'w_init': parameters.weights,
               'ddf': P.ddf}
stdp_params.update(P.timing_dependence.params)
stdp_params.update(P.weight_dependence.params)
S = STDPSynapse(parameters.delays,
                parameters.input_spike_times,
                spike_data[sim_list[0].__name__]['post'][0].spike_times,
                **stdp_params)

key = "pre→post"

def plot_figures():
    import pylab
    pylab.rcParams['interactive'] = True
    
    # plot Vm
    pylab.figure(1)
    for sim_name, vm in vm_data.items():
        pylab.plot(vm['post'][0].time_axis(), vm['post'][0].signal, label="post (%s)" % sim_name)
    pylab.legend(loc='upper left')
    pylab.xlim(0, sim_time)
    pylab.title("Vm")
    
    # plot spikes
    pylab.figure(2)
    for i, (sim_name, spikes) in enumerate(spike_data.items()):
        pylab.plot( spikes['post'][0].spike_times, i*numpy.ones_like(spikes['post'][0].spike_times),
                   "|", label="Postsynaptic spikes (%s)" % sim_name, markersize=100)
    pylab.plot( parameters.input_spike_times, (i+1)*numpy.ones_like(parameters.input_spike_times),
               "|", label="Presynaptic spikes", markersize=100 )
    pylab.xlim(0, sim_time)
    pylab.ylim(-0.5,i+1.5)
    pylab.title("Spikes (red=presynaptic)")
    
    # plot weights
    key = "pre→post"
    pylab.figure(3)
    for sim_name, w in w_data.items():
        pylab.plot(w[key][:,0], w[key][:,1], label="%s (%s)" % (key, sim_name))
    t,w = S.calc_weights(at_input_spiketimes=True)
    pylab.plot(t, w, label="%s (calculated)" % key)
    pylab.xlim(0, sim_time)
    pylab.legend(loc='upper left')
    pylab.title("Weights at input spike times")
    
    pylab.figure(4)
    for sim_name, w in w_data1.items():
        pylab.plot(w[key][:,0], w[key][:,1], label="%s (%s)" % (key, sim_name))
    t,w = S.calc_weights(at_input_spiketimes=False)
    pylab.plot(t, w, label="%s (calculated)" % key)
    pylab.xlim(0, sim_time)
    pylab.legend(loc='upper left')
    pylab.title("Weights at all spike times")

if PLOT_FIGURES:
    plot_figures()

for sim_name, w in w_data.items():
    print sim_name, w[key][:,1]

def diff(data):
    arr = [x[key][:,1] for x in data.values()]
    a = arr[0]
    sum = 0
    for b in arr[1:]:
        assert len(a) == len(b)
        sum += (numpy.sum(abs(a - b)) / ((a.mean()+b.mean())/2))
    return sum/len(a)

print "seed was", seed
print "difference", diff(w_data)
