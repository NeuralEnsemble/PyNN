"""
A single IF neuron with alpha-function shaped, current-based synapses, fed by a
single spike source.

Andrew Davison, UNIC, CNRS
May 2006

$Id$
"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)


id = setup(timestep=0.01,min_delay=0.1)

ifcells = create(IF_curr_alpha(i_offset=-0.1, tau_refrac=0.1, v_thresh=52.2), n=5)

spike_source = create(SpikeSourceArray(spike_times=[0.1*float(i) for i in range(1,1001,1)]))

conn = connect(spike_source,ifcells,weight=1.5)

record('v', ifcells[0:1], "Results/IF_curr_alpha2_%s.pkl" % simulator_name)
run(100.0)

end()
