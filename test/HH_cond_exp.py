"""
Simple test of HH_cond_exp standard model

Andrew Davison, UNIC, CNRS
July 2007

$Id:$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.01,min_delay=0.01,max_delay=4.0)

hhcell = create(HH_cond_exp)

spike_sourceE = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(5,105,10)]})
spike_sourceI = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(100,255,10)]})
 
connE = connect(spike_sourceE, hhcell, weight=1.5, synapse_type='excitatory', delay=2.0)
connI = connect(spike_sourceI, hhcell, weight=-1.5, synapse_type='inhibitory', delay=4.0)
    
record_v(hhcell, "HH_cond_exp_%s.v" % simulator)
run(200.0)

end()

