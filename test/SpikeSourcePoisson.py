"""
Simple test of recording a SpikeSourcePoisson object

Andrew Davison, UNIC, CNRS
September 2006

$Id: IF_curr_exp.py 43 2007-05-10 14:06:33Z apdavison $
"""

import sys

simulator_name = sys.argv[-1]

exec("from pyNN.%s import *" % simulator_name)


setup(timestep=0.1, min_delay=0.1)

poissonsource = create(SpikeSourcePoisson,{'rate' : 100., 'duration' : 100., 'start' : 100.})

record(poissonsource, "Results/SpikeSourcePoisson_%s.ras" % simulator_name)
run(300.0)
  
end()
