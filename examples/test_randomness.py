# coding: utf-8
"""
An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The IF network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).

Usage: python VAbenchmarks.py <simulator> <benchmark>

    <simulator> is either neuron, nest, brian or pcsim
    <benchmark> is either CUBA or COBA.

Andrew Davison, UNIC, CNRS
August 2006

"""

import os
import socket
from math import *

from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename
sim, options = get_simulator(("benchmark", "Either CUBA or COBA"),
			     ("--plot-figure", "Plot the simulation results to a file."))

from pyNN.random import NumpyRNG, RandomDistribution
from numpy import nan_to_num

init_logging(None, debug=True)
timer = Timer()

# === Define parameters ========================================================

threads  = 1
rngseed  = 98765
parallel_safe = True

n        = 400  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.2  # connection probability
stim_dur = 60.   # (ms) duration of random stimulation
rate     = 20.  # (Hz) frequency of the random stimulation

dt       = 0.1   # (ms) simulation timestep
tstop    = 200 #1000  # (ms) simulaton duration
delay    = 0.2

# Cell parameters
area     = 20000. # (µm²)
tau_m    = 20.    # (ms)
cm       = 1.     # (µF/cm²)
g_leak   = 5e-5   # (S/cm²)
if options.benchmark == "COBA":
    E_leak   = -50.  # (mV)
elif options.benchmark == "CUBA":
    E_leak   = -49.  # (mV)
v_thresh = -45.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 2.5     # (ms)
tau_inh  = 5.    # (ms)

# Synapse parameters
if options.benchmark == "COBA":
    Gexc = 4.     # (nS)
    Ginh = 51.    # (nS)
elif options.benchmark == "CUBA":
    Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
    Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -100.   # (mV)

### what is the synaptic delay???

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells
n_inh = n - n_exc                     # number of inhibitory cells
if options.benchmark == "COBA":
    celltype = sim.IF_cond_exp
    w_exc    = Gexc*1e-3              # We convert conductances to uS
    w_inh    = Ginh*1e-3
elif options.benchmark == "CUBA":
    celltype = sim.IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
    assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

extra = {'loglevel':2, 'useSystemSim':True,
	'maxNeuronLoss':0., 'maxSynapseLoss':0.4,
	'hardwareNeuronSize':8,
	'threads' : threads,
	'filename': "va_%s.xml" % options.benchmark,
	'label': 'VA'}
if sim.__name__ == "pyNN.hardware.brainscales":
  extra['hardware'] = sim.hardwareSetup['small']
  
if options.simulator == "neuroml":
    extra["file"] = "VAbenchmarks.xml"

node_id = sim.setup(timestep=dt, min_delay=delay, max_delay=1.0, **extra)
np = sim.num_processes()

host_name = socket.gethostname()
print "Host #%d is on %s" % (node_id+1, host_name)

print "%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads'])

cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

if (options.benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh

timer.start()

print "%s Creating cell populations..." % node_id
exc_cells = sim.Population(n_exc, celltype(**cell_params), label="Excitatory_Cells")
inh_cells = sim.Population(n_inh, celltype(**cell_params), label="Inhibitory_Cells")

rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

if options.benchmark == "COBA":
    spike_times = [float(i) for i in range(50,int(50+stim_dur),int(1000./rate))]
    ext_stim = sim.Population(20, sim.SpikeSourceArray(spike_times = spike_times), label="spikes")
    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn, rng=rng)
    ext_syn = sim.StaticSynapse(weight=0.1)

print "%s Initialising membrane potential to random values..." % node_id
#uniformDistr = RandomDistribution('uniform', [v_reset,v_thresh], rng=rng)
exc_cells.initialize(v=E_leak)
inh_cells.initialize(v=E_leak)

print "%s Connecting populations..." % node_id
progress_bar = ProgressBar(width=20)
conn = sim.FixedProbabilityConnector(pconn, rng=rng, callback=progress_bar)
exc_syn = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_syn = sim.StaticSynapse(weight=w_inh, delay=delay)

connections={}
connections['e2e'] = sim.Projection(exc_cells, exc_cells, conn, exc_syn, receptor_type='excitatory')
connections['e2i'] = sim.Projection(exc_cells, inh_cells, conn, exc_syn, receptor_type='excitatory')
connections['i2e'] = sim.Projection(inh_cells, exc_cells, conn, inh_syn, receptor_type='inhibitory')
connections['i2i'] = sim.Projection(inh_cells, inh_cells, conn, inh_syn, receptor_type='inhibitory')
if (options.benchmark == "COBA"):
    connections['ext2e'] = sim.Projection(ext_stim, exc_cells, connector=ext_conn, synapse_type=ext_syn, receptor_type='excitatory')
    connections['ext2i'] = sim.Projection(ext_stim, inh_cells, connector=ext_conn, synapse_type=ext_syn, receptor_type='excitatory')

print nan_to_num(connections['ext2e'].get('delay', format="array"))[0:10,0:10]
print nan_to_num(connections['ext2i'].get('delay', format="array"))[0:10,0:10]

# === Setup recording ==========================================================
print "%s Setting up recording..." % node_id
exc_cells.record('spikes')
inh_cells.record('spikes')
exc_cells[0, 1].record('v')
inh_cells[0, 1].record('v')

buildCPUTime = timer.diff()

# === Save connections to file =================================================

#for prj in connections.keys():
    #connections[prj].saveConnections('Results/VAbenchmark_%s_%s_%s_np%d.conn' % (benchmark, prj, options.simulator, np))
saveCPUTime = timer.diff()

# === Run simulation ===========================================================
print "%d Running simulation..." % node_id

sim.run(tstop)

simCPUTime = timer.diff()

E_count = exc_cells.mean_spike_count()
I_count = inh_cells.mean_spike_count()

# === Print results to file ====================================================

print "%d Writing data to file..." % node_id

filename_exc = normalized_filename("Results", "VAbenchmarks_%s_exc" % options.benchmark, "pkl",
                        options.simulator, np)
exc_cells.write_data(
    filename_exc,
    annotations={'script_name': __file__})

filename_inh = normalized_filename("Results", "VAbenchmarks_%s_inh" % options.benchmark, "pkl",
                        options.simulator, np)
inh_cells.write_data(
    filename_inh,
    annotations={'script_name': __file__})
writeCPUTime = timer.diff()

str_connections = "%d e→e  %d e→i  %d i→e  %d i→i" % (connections['e2e'].size(),
                                                  connections['e2i'].size(),
                                                  connections['i2e'].size(),
                                                  connections['i2i'].size())
str_stim_connections = "%d stim->e  %d stim->i" % (connections['ext2e'].size(),connections['ext2i'].size())

if node_id == 0:
    print "\n--- Vogels-Abbott Network Simulation ---"
    print "Nodes                  : %d" % np
    print "Simulation type        : %s" % options.benchmark
    print "Number of Neurons      : %d" % n
    print "Number of Synapses     : %s" % str_connections
    print "Number of inputs       : %s" % str_stim_connections
    print "Excitatory conductance : %g nS" % Gexc
    print "Inhibitory conductance : %g nS" % Ginh
    print "Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,)
    print "Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,)
    print "Build time             : %g s" % buildCPUTime
    print "Save connections time  : %g s" % saveCPUTime
    print "Simulation time        : %g s" % simCPUTime
    print "Writing time           : %g s" % writeCPUTime
    print "\n--- files ---"
    print filename_exc
    print filename_inh

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    data_exc = exc_cells.get_data().segments[0]
    vm_exc = data_exc.filter(name="v")[0]
    data_inh = inh_cells.get_data().segments[0]
    vm_inh = data_inh.filter(name="v")[0]
    matrix_weights_exc = nan_to_num(connections['ext2e'].get('weight', format="array"))[0:20,0:60]
    matrix_weights_inh = nan_to_num(connections['ext2i'].get('weight', format="array"))[0:20,0:60]
    conn_weights_exc = nan_to_num(connections['e2e'].get('weight', format="array"))[0:20,0:60]
    conn_weights_inh = nan_to_num(connections['e2i'].get('weight', format="array"))[0:20,0:60]
    inh_weights_exc = nan_to_num(connections['i2e'].get('weight', format="array"))[0:20,0:60]
    inh_weights_inh = nan_to_num(connections['i2i'].get('weight', format="array"))[0:20,0:60]
 
    Figure(
        #Panel(vm_exc, ylabel="Membrane potential (mV)", data_labels=["excitatory", "excitatory"], line_properties=[{'ylim':[-70, -40]}, {'ylim':[-70, -40]}]),
        #Panel(data_exc.spiketrains[0:60], xlabel="Time (ms)", xticks=True),
        #Panel(vm_inh, ylabel="Membrane potential (mV)", data_labels=["inhibitory", "inhibitory"], line_properties=[{'ylim':[-70, -40]}, {'ylim':[-70, -40]}]),
        #Panel(data_inh.spiketrains[0:60], xlabel="Time (ms)", xticks=True),
        Panel(matrix_weights_exc,data_labels=["ext->exc"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        Panel(matrix_weights_inh,data_labels=["ext->inh"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        Panel(conn_weights_exc,data_labels=["exc->exc"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        Panel(conn_weights_inh,data_labels=["exc->inh"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        Panel(inh_weights_exc,data_labels=["inh->exc"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
        Panel(inh_weights_inh,data_labels=["inh->inh"], line_properties=[{'xticks':True, 'yticks':True, 'cmap':'Greys'}]),
     ).save(options.plot_figure)

# === Finished with simulator ==================================================

sim.end()
