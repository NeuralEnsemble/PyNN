"""

$Id$
"""

import sys
from NeuroTools.stgen import StGen
from pyNN.random import NumpyRNG

simulator_name = sys.argv[-1]

exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.1, min_delay=2.0, max_delay=4.0)

ifcell = create(IF_curr_exp,{'i_offset' :   0.1, 'tau_refrac' : 3.0,
                             'v_thresh' : -51.0, 'tau_syn_E'  : 2.0,
                             'tau_syn_I':  5.0,  'v_reset'    : -70.0,
                             'v_init'   : -53.2})
input_rate = 200.0
simtime = 1000.0
seed = 240965239

spike_generator = StGen(numpyrng=NumpyRNG(seed=seed))

spike_source = create(SpikeSourceArray,
                      {'spike_times': spike_generator.poisson_generator((input_rate/1000.0),simtime)}) # rate in spikes/ms)

 
conn = connect(spike_source, ifcell, weight=1.5, synapse_type='excitatory', delay=2.0)
    
record(ifcell, "Results/IF_curr_exp2_%s.ras" % simulator_name)
record_v(ifcell, "Results/IF_curr_exp2_%s.v" % simulator_name)
run(simtime)
  
end()
