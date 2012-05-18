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
import glob
from pyNN.recording import get_io

plt.ion() #.rcParams['interactive'] = True

example = sys.argv[1]

blocks = {}
for simulator in 'PCSIM', 'NEST', 'NEURON', 'Brian', 'MOOSE', 'Nemo':
    pattern = "Results/%s_*%s.*" % (example, simulator.lower())
    datafiles = glob.glob(pattern)
    if datafiles:
        for datafile in datafiles:
            base = os.path.basename(datafile)
            root = base[:base.find(simulator.lower())-1]
            if root not in blocks:
                blocks[root] = {}
            blocks[root][simulator] = get_io(datafile).read_block()
    else:
        print "No data found for pattern %s" % pattern

print "-"*79
print example
from pprint import pprint
pprint(blocks)

if len(blocks) > 0:
    for name in blocks:
        plt.figure()
        lw = 2*len(blocks[name]) - 1
        for simulator, block in blocks[name].items():
            vm = block.segments[0].filter(name="v")[0]
            plt.plot(vm.times, vm[:, 0], label=simulator, linewidth=lw)
            lw -= 2
        plt.legend()
        plt.title(name)
        plt.xlabel("Time (ms)")
        plt.ylabel("Vm (mV)")

        plt.savefig("Results/%s.png" % name)

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