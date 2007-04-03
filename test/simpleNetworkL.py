"""
Simple network, using only the low-level interface,
with a 1D population of poisson spike sources
projecting to a 2D population of IF_curr_alpha neurons.

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

import sys
from NeuroTools.stgen import StGen

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)

tstop = 1000.0 # all times in milliseconds

setup(timestep=0.025,min_delay=0.025)

cell_params = {'tau_refrac':2.0,'v_thresh':-50.0,'tau_syn':2.0}
ifcell1 = create(IF_curr_alpha, cell_params)
ifcell2 = create(IF_curr_alpha, cell_params)

spikeGenerator = StGen()
spike_times = list(spikeGenerator.poisson_generator(100.0/1000.0,tstop)) # rate in spikes/ms

spike_source = create(SpikeSourceArray, {'spike_times': spike_times })
 
conn1 = connect(spike_source, ifcell1, weight=1.0)
conn2 = connect(spike_source, ifcell2, weight=1.0)
    
record_v(ifcell1,"simpleNetworkL_%s_1.v" % simulator)
record_v(ifcell2,"simpleNetworkL_%s_2.v" % simulator)
run(tstop)
    
end()

