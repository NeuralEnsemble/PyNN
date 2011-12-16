# coding: utf-8
"""
An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The IF network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).

This version demonstrates an alternative way to set-up the network, using
PopulationViews.

Andrew Davison, UNIC, CNRS
March 2010

$Id: $
"""

import os
import socket
from math import *
import csa

from pyNN.utility import get_script_args, Timer
usage = """Usage: python VAbenchmarks.py <simulator> <benchmark>
           <simulator> is either neuron, nest, brian or pcsim
           <benchmark> is either CUBA or COBA."""
simulator_name, benchmark = get_script_args(2, usage)  
exec("from pyNN.%s import *" % simulator_name)
from pyNN.random import NumpyRNG, RandomDistribution

timer = Timer()

# === Define parameters ========================================================

threads  = 1
rngseed  = 98765
parallel_safe = True

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 50.   # (ms) duration of random stimulation
rate     = 100.  # (Hz) frequency of the random stimulation

dt       = 0.1   # (ms) simulation timestep
tstop    = 1000  # (ms) simulaton duration
delay    = 0.2

# Cell parameters
area     = 20000. # (µm²)
tau_m    = 20.    # (ms)
cm       = 1.     # (µF/cm²)
g_leak   = 5e-5   # (S/cm²)
if benchmark == "COBA":
    E_leak   = -60.  # (mV)
elif benchmark == "CUBA":
    E_leak   = -49.  # (mV)
v_thresh = -50.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms)

# Synapse parameters
if benchmark == "COBA":
    Gexc = 4.     # (nS)
    Ginh = 51.    # (nS)
elif benchmark == "CUBA":
    Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
    Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

### what is the synaptic delay???

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells   
n_inh = n - n_exc                     # number of inhibitory cells
if benchmark == "COBA":
    celltype = IF_cond_exp
    w_exc    = Gexc*1e-3              # We convert conductances to uS
    w_inh    = Ginh*1e-3
elif benchmark == "CUBA":
    celltype = IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
    assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

extra = {'threads' : threads}
if simulator_name == "neuroml":
    extra["file"] = "VAbenchmarks.xml"

node_id = setup(timestep=dt, min_delay=delay, max_delay=delay, **extra)
np = num_processes()

host_name = socket.gethostname()
print "Host #%d is on %s" % (node_id+1, host_name)

print "%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads'])
    
cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

if (benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh
    
timer.start()

print "%s Creating cell populations..." % node_id
all_cells = Population(n_exc+n_inh, celltype, cell_params, label="All_Cells")
exc_cells = all_cells[:n_exc]
inh_cells = all_cells[n_exc:]
if benchmark == "COBA":
    ext_stim = Population(20, SpikeSourcePoisson, {'rate' : rate, 'duration' : stim_dur}, label="expoisson")
    rconn = 0.01
    ext_conn = FixedProbabilityConnector(rconn, weights=0.1)

print "%s Initialising membrane potential to random values..." % node_id
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', [v_reset,v_thresh], rng=rng)
all_cells.initialize('v', uniformDistr)

print "%s Connecting populations..." % node_id
#exc_conn = FixedProbabilityConnector(pconn, weights=w_exc, delays=delay)
#inh_conn = FixedProbabilityConnector(pconn, weights=w_inh, delays=delay)
exc_conn = CSAConnector(csa.cset (csa.random (pconn), w_exc, delay))
inh_conn = CSAConnector(csa.cset (csa.random (pconn), w_inh, delay))

connections={}
connections['exc'] = Projection(exc_cells, all_cells, exc_conn, target='excitatory', rng=rng)
connections['inh'] = Projection(inh_cells, all_cells, inh_conn, target='inhibitory', rng=rng)
if (benchmark == "COBA"):
    connections['ext'] = Projection(ext_stim, all_cells, ext_conn, target='excitatory')

# === Setup recording ==========================================================
print "%s Setting up recording..." % node_id
all_cells.record()
vrecord_list = [exc_cells[0],exc_cells[1]]
exc_cells.record_v(vrecord_list)

buildCPUTime = timer.diff()

# === Save connections to file =================================================

#print "%s Saving connections to file..." % node_id
#for prj in connections.keys():
#    connections[prj].saveConnections('Results/VAbenchmark_%s_%s_%s_np%d.conn' % (benchmark, prj, simulator_name, np))
#saveCPUTime = timer.diff()

# === Run simulation ===========================================================
print "%d Running simulation..." % node_id

run(tstop)

simCPUTime = timer.diff()

E_count = exc_cells.mean_spike_count()
I_count = inh_cells.mean_spike_count()

# === Print results to file ====================================================

print "%d Writing data to file..." % node_id

if not(os.path.isdir('Results')):
    os.mkdir('Results')

exc_cells.printSpikes("Results/VAbenchmark_%s_exc_%s_np%d.ras" % (benchmark, simulator_name, np))
inh_cells.printSpikes("Results/VAbenchmark_%s_inh_%s_np%d.ras" % (benchmark, simulator_name, np))
exc_cells.print_v("Results/VAbenchmark_%s_exc_%s_np%d.v" % (benchmark, simulator_name, np))
writeCPUTime = timer.diff()

connections = "%d e→e,i  %d i→e,i" % (connections['exc'].size(),
                                      connections['inh'].size(),)

if node_id == 0:
    print "\n--- Vogels-Abbott Network Simulation ---"
    print "Nodes                  : %d" % np
    print "Simulation type        : %s" % benchmark
    print "Number of Neurons      : %d" % n
    print "Number of Synapses     : %s" % connections
    print "Excitatory conductance : %g nS" % Gexc
    print "Inhibitory conductance : %g nS" % Ginh
    print "Excitatory rate        : %g Hz" % (E_count*1000.0/tstop,)
    print "Inhibitory rate        : %g Hz" % (I_count*1000.0/tstop,)
    print "Build time             : %g s" % buildCPUTime
    #print "Save connections time  : %g s" % saveCPUTime
    print "Simulation time        : %g s" % simCPUTime
    print "Writing time           : %g s" % writeCPUTime


# === Finished with simulator ==================================================

end()
