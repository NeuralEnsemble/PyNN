"""
Simple network, using only the low-level interface, with a Poisson spike source
projecting to a pair of IF_curr_alpha neurons.

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

import numpy
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)

tstop = 1000.0 # all times in milliseconds
rate = 100.0 # spikes/s

setup(timestep=0.1,min_delay=0.2)

cell_params = {'tau_refrac':2.0, 'v_thresh':-50.0, 'tau_syn_E':2.0, 'tau_syn_I' : 4.0}
ifcell1 = create(IF_curr_alpha, cell_params)
ifcell2 = create(IF_curr_alpha, cell_params)

number = int(2*tstop*rate/1000.0)
numpy.random.seed(637645386)
spike_times = numpy.add.accumulate(numpy.random.exponential(1000.0/rate, size=number))
assert spike_times.max() > tstop

spike_source = create(SpikeSourceArray, {'spike_times': spike_times })
 
conn1 = connect(spike_source, ifcell1, weight=1.0)
conn2 = connect(spike_source, ifcell2, weight=1.0)
    
record('v', ifcell1+ifcell2, "Results/simpleNetworkL_%s_v.pkl" % simulator_name)
run(tstop)
    
end()

