"""
Plot graphs for the example scripts in this directory.

Run with:

python plot_results.py <example_name>

where <example_name> is the first part of the example script name, without the ".py"

e.g. python plot_results.py IF_curr_exp

"""

import sys
import os
import matplotlib.pyplot as plt
import numpy
import warnings
from pyNN.recording import get_io
pylab.rcParams['interactive'] = True

example = sys.argv[1]

blocks = {}
for simulator in 'PCSIM', 'NEST', 'NEURON', 'Brian':
    datafiles = glob.glob("Results/%s_*_%s.*" % (example, simulator.lower()))
    assert len(datafiles) == 1
    blocks[simulator] = get_io(datafiles[0]).read()

for simulator, block in blocks.items():
    vm = block.segments[0].filter(name="v")
    plt.plot(vm.times, vm[:, 0], label=simulator)
plt.legend()
plt.title(example)
plt.xlabel("Time (ms)")
plt.ylabel("Vm (mV)")

plt.savefig("Results/%s.png" % example)

#if gsyn_data:
#    pylab.figure(2)
#    for label, gsyn in gsyn_data.items():
#        pylab.plot(t, gsyn[:,0], label="%s (exc)" % label)
#        pylab.plot(t, gsyn[:,1], label="%s (inh)" % label)
#        pylab.legend()
#        pylab.title(example)
#        pylab.xlabel("Time (ms)")
#        pylab.ylabel("Synaptic conductance (nS)")
#        pylab.savefig("Results/%s_gsyn.png" % example)