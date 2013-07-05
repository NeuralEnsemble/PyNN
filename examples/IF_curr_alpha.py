"""
A single IF neuron with alpha-function shaped, current-based synapses, fed by
two spike sources.

Run as:

$ python IF_curr_alpha.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
May 2006

"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.1,min_delay=0.1,max_delay=4.0)

ifcell = create(IF_curr_alpha(i_offset=0.1, tau_refrac=3.0, v_thresh=-51.0,
                              tau_syn_E=2.0, tau_syn_I=5.0, v_reset=-70.0))

spike_sourceE = create(SpikeSourceArray(spike_times=[float(i) for i in range(5,105,10)]))
spike_sourceI = create(SpikeSourceArray(spike_times=[float(i) for i in range(100,255,10)]))

connE = connect(spike_sourceE, ifcell, weight=1.5, receptor_type='excitatory', delay=2.0)
connI = connect(spike_sourceI, ifcell, weight=-1.5, receptor_type='inhibitory', delay=4.0)

record('v', ifcell, "Results/IF_curr_alpha_%s.pkl" % simulator_name)

initialize(ifcell, v=-54.2)

run(200.0)

end()
