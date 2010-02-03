"""
Plot graphs for the example scripts in this directory.

Run with:

python plot_results.py <example_name> [<dt>]

where <example_name> is the first part of the example script name, without the
".py" and <dt> (optional) is the timestep used in the example (defaults to 0.1).

e.g. python plot_results.py IF_curr_exp 0.1

"""

import sys
import os
import pylab
import numpy
import warnings
pylab.rcParams['interactive'] = True

example = sys.argv[1]
if len(sys.argv) > 2:
    dt = float(sys.argv[2])
else:
    dt = 0.1

vm_data = {}
gsyn_data = {}
for simulator in 'PCSIM', 'NEST', 'NEURON', 'Brian':
    datafile = "Results/%s_%s.v" % (example, simulator.lower())
    if os.path.exists(datafile):
        data = numpy.loadtxt(datafile)
        first_id = data[:,1].min()
        data0 = data[data[:,1]==first_id] # if there are multiple cells, we just plot the first
        vm_data[simulator] = data0
    datafile = "Results/%s_%s.gsyn" % (example, simulator.lower())
    if os.path.exists(datafile):
        gsyn_data[simulator] = numpy.loadtxt(datafile)



sizes = pylab.array([vm.size for vm in vm_data.values()])
if not all(sizes == sizes[0]):
#if not pcsim_data.shape == nest_data.shape == neuron_data.shape:
    errmsg = "Data has different lengths. " + ", ".join(["%s: %s" % (simulator, vm.shape) for simulator, vm in vm_data.items()])
    errmsg += "Trimming to the length of the shortest."
    warnings.warn(errmsg)
    new_length = min([vm.shape[0] for vm in vm_data.values()])
    for simulator in vm_data:
        vm_data[simulator] = vm_data[simulator][:new_length,:]
    for simulator in gsyn_data:
        gsyn_data[simulator] = gsyn_data[simulator][:new_length,:]
    

t = dt*pylab.arange(vm_data[vm_data.keys()[0]].shape[0])

for label, vm in vm_data.items():
    pylab.plot(t, vm[:,0], label=label)
pylab.legend()
pylab.title(example)
pylab.xlabel("Time (ms)")
pylab.ylabel("Vm (mV)")

pylab.savefig("Results/%s.png" % example)

if gsyn_data:
    pylab.figure(2)
    for label, gsyn in gsyn_data.items():
        pylab.plot(t, gsyn[:,0], label="%s (exc)" % label)
        pylab.plot(t, gsyn[:,1], label="%s (inh)" % label)
        pylab.legend()
        pylab.title(example)
        pylab.xlabel("Time (ms)")
        pylab.ylabel("Synaptic conductance (nS)")
        pylab.savefig("Results/%s_gsyn.png" % example)