"""
A single-compartment Hodgkin-Huxley neuron with exponential, conductance-based
synapses, fed by a current injection.

Run as:

$ python HH_cond_exp2.py <simulator>

where <simulator> is 'neuron', 'nest', etc

Andrew Davison, UNIC, CNRS
March 2010

$Id:$
"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.001, min_delay=0.1)

hhcell = create(HH_cond_exp, cellparams={'i_offset': 1.0})
    
record_v(hhcell, "Results/HH_cond_exp2_%s.v" % simulator_name)

run(10.0)

#import pylab
#pylab.rcParams['interactive'] = True
#v = pylab.array(hhcell._cell.vmTable)/mV
#t = get_time_step()*pylab.arange(0.0, v.size)
#pylab.plot(t, v)
#pylab.xlabel("time (ms)")
#pylab.ylabel("Vm (mV)")

end()

