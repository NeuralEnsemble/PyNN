"""
A single-compartment Hodgkin-Huxley neuron with exponential, conductance-based
synapses, fed by two spike sources.

Run as:

$ python HH_cond_exp.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
July 2007

"""

from pyNN.utility import get_script_args, normalized_filename


simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.01, min_delay=0.1, max_delay=4.0, quit_on_end=False)

hhcell = create(HH_cond_exp())

spike_sourceE = create(SpikeSourceArray(spike_times=[float(i) for i in range(1,100,1)]))
spike_sourceI = create(SpikeSourceArray(spike_times=[float(i) for i in range(100,200,11)]))

connE = connect(spike_sourceE, hhcell, weight=0.02, receptor_type='excitatory', delay=2.0)
connI = connect(spike_sourceI, hhcell, weight=0.01, receptor_type='inhibitory', delay=4.0)

filename = normalized_filename("Results", "HH_cond_exp", "pkl", simulator_name)
record(['v', 'gsyn_exc', 'gsyn_inh'], hhcell, filename,
       annotations={'script_name': __file__})

run(200.0)

end()
