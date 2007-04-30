"""
Simple test to compare iaf_exp_neuron2 in PyNEST with StandardIF in NEURON.

Andrew Davison, UNIC, CNRS
September 2006

$Id$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "oldneuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.01,min_delay=0.01)

ifcell = create(IF_curr_exp,{'i_offset':   0.1, 'tau_refrac': 5.0,
                               'v_thresh': -51.0, 'tau_syn_E':  2.0,
                               'tau_syn_I':  5.0, 'v_reset':  -70.0})

spike_sourceE = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(5,105,10)]})
spike_sourceI = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(105,205,10)]})
 
connE = connect(spike_sourceE,ifcell,weight=1.5)
connI = connect(spike_sourceI,ifcell,weight=-1.5)
    
record_v(ifcell,"IF_curr_exp_%s.v" % simulator)
run(100.0)
  
    
end()