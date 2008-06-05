"""

$Id: IF_curr_exp.py 97 2007-06-07 12:09:56Z Pierre $
"""

import sys
from NeuroTools.stgen import StGen
from pyNN.random import NumpyRNG

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.01, min_delay=2.0, max_delay=4.0)

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
    
record(ifcell,"IF_curr_exp2_%s.ras" % simulator)
record_v(ifcell,"IF_curr_exp2_%s.v" % simulator)
run(simtime)
  
end()