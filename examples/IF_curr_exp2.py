"""
A single IF neuron with exponential, current-based synapses, fed by a single
Poisson spike source.

Run as:

$ python IF_curr_exp2.py <simulator>

where <simulator> is 'neuron', 'nest', etc

"""

import numpy
from pyNN.utility import get_script_args
simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)
from pyNN.random import NumpyRNG

setup(timestep=0.01, min_delay=2.0, max_delay=4.0)

ifcell = create(IF_curr_exp,{'i_offset' :   0.1, 'tau_refrac' : 3.0,
                             'v_thresh' : -51.0, 'tau_syn_E'  : 2.0,
                             'tau_syn_I':  5.0,  'v_reset'    : -70.0})
input_rate = 200.0
simtime = 1000.0
seed = 240965239

rng = NumpyRNG(seed=seed)
n_spikes = input_rate*simtime/1000.0
spike_times = numpy.add.accumulate(rng.next(n_spikes, 'exponential', {'beta': 1000.0/input_rate}))

spike_source = create(SpikeSourceArray(spike_times=spike_times))


conn = connect(spike_source, ifcell, weight=1.5, receptor_type='excitatory', delay=2.0)

record(('spikes', 'v'), ifcell, "Results/IF_curr_exp2_%s.pkl" % simulator_name)
initialize(ifcell, v=-53.2)

run(simtime)

end()
