"""
Simple test of recording a SpikeSourcePoisson object

Andrew Davison, UNIC, CNRS
September 2006

$Id$
"""

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.01, min_delay=0.1)

poissonsource = Population(10, SpikeSourcePoisson, {'rate' : 100.0, 'duration' : 100.0, 'start' : 100.0})
poissonsource.record()

run(300.0)
  
print "Mean spike count:", poissonsource.mean_spike_count()
print "First few spikes:"
all_spikes = poissonsource.getSpikes()
first_id = all_spikes[0,0]
for i, cell in enumerate(poissonsource):
    print "cell #%d: %s" % (cell, all_spikes[all_spikes[:,0]==i][:,1][:5])
  
poissonsource.printSpikes("Results/SpikeSourcePoisson_%s.ras" % simulator_name)
  
end()
