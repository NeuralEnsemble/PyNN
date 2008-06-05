"""
Simple test of recording a SpikeSourcePoisson object

Andrew Davison, UNIC, CNRS
September 2006

$Id: IF_curr_exp.py 43 2007-05-10 14:06:33Z apdavison $
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.01, min_delay=0.01)

poissonsource = create(SpikeSourcePoisson,{'rate' : 100., 'duration' : 100., 'start' : 100.})

record(poissonsource,"SpikeSourcePoisson_%s.ras" % simulator)
run(300.0)
  
end()
