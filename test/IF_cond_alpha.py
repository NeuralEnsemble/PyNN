"""
Simple test to compare iaf_neuron in NEST with StandardIF in NEURON.

Andrew Davison, UNIC, CNRS
May 2006

$Id$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.025,min_delay=0.025)

ifcell = create(IF_cond_alpha, {'i_offset' : 0.,     'tau_refrac' : 5.0,
                                'v_thresh' : -51.0,  'tau_syn_E'  : 5.0,
                                'tau_syn_I': 10.0,    'v_reset'    : -70.0,
                                 'e_rev_E' : 0.,     'e_rev_I'    : -80.})

spike_source = create(SpikeSourceArray, {'spike_times': [float(i) for i in range(5,105,10)]})

connect(spike_source,ifcell,weight=0.006,synapse_type='excitatory')

### Inhibitory synapses not taken into account in NEST ??
#connect(spike_source,ifcell,weight=0.067,synapse_type='inhibitory')
    
record_v(ifcell,"IF_cond_alpha_%s.v" % simulator)
run(100.0)

end()

