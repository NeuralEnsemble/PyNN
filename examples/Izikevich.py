"""
A single IF neuron with exponential, current-based synapses, fed by two
spike sources.

Run as:

$ python IF_curr_exp.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
September 2006

$Id: IF_curr_exp.py 756 2010-05-18 12:57:19Z apdavison $
"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)

setup(timestep=1., min_delay=1., max_delay=4.0)

ifcell = create(Izikevich , {'a' : 0.015, 'd' : 1.5})

spike_sourceE = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(5,105,10)]})
#spike_sourceE = create(SpikeSourcePoisson, {'rate': 100.})
spike_sourceI = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(155,255,10)]})
 
connE = connect(spike_sourceE, ifcell, weight=1.5, synapse_type='excitatory', delay=2.0)
connI = connect(spike_sourceI, ifcell, weight=-1.5, synapse_type='inhibitory', delay=4.0)
record_v(ifcell, "Results/Izikevich_%s.v" % simulator_name)
initialize(ifcell, 'v', -53.2)
run(200)
  
end()
