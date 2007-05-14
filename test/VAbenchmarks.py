# coding: utf-8
"""

This is an attempt to implement the FACETS review paper benchmarks 1 and 2:

The IF network is based on the CUBA and COBA models of Vogels & Abbott (J
Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).

NOTE: the benchmark specifies an initial 50 ms external spike input, but
this has not been implemented (not necessary for sustained firing).

Andrew Davison, UNIC, CNRS
August 2006

$Id:VAbenchmarks.py 5 2007-04-16 15:01:24Z davison $
"""

import sys
from copy import copy
from NeuroTools.stgen import StGen
if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "oldneuron"    # run using nrngui -python
exec("from pyNN.%s import *" % simulator)

from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.utility

# === Define parameters ========================================================

benchmark = "CUBA"
rngseed  = 98765

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 50    # (ms) duration of random stimulation

dt       = 0.01  # (ms) simulation timestep
tstop    = 4000  # (ms) simulaton duration

# Cell parameters
area     = 20000 # (µm²)
tau_m    = 20    # (ms)
cm       = 1     # (µF/cm²)
g_leak   = 5e-5  # (S/cm²)
if benchmark == "COBA":
    E_leak   = -60   # (mV)
elif benchmark == "CUBA":
    E_leak   = -49   # (mV)
v_thresh = -50   # (mV)
v_reset  = -60   # (mV)
t_refrac = 5     # (ms) (clamped at v_reset)
v_mean   = -60   # (mV) 'mean' membrane potential, for calculating CUBA weights

# Synapse parameters
if benchmark == "COBA":
    Gexc = 6.0   # (nS)
    Ginh = 27.0  # (nS)
elif benchmark == "CUBA":
    Gexc = 0.27  # (nS)
    Ginh = 4.5   # (nS)
Erev_exc = 0     # (mV)
Erev_inh = -80   # (mV)
tau_exc  = 5     # (ms)
tau_inh  = 10    # (ms)

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
    w_exc = Gexc
    w_inh = Ginh
elif benchmark == "CUBA":
    celltype = IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
    assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

node_id = setup(timestep=dt,min_delay=0.1,max_delay=0.1)
#if simulator=='nest':
#    pynest.showNESTStatus()

cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

Timer.start()

print "%d Creating cell populations..." % node_id
exc_cells = Population((n_exc,), celltype, cell_params, "Excitatory_Cells")
inh_cells = Population((n_inh,), celltype, cell_params, "Inhibitory_Cells")

print "%d Initialising membrane potential to random values..." % node_id
rng = NumpyRNG(rngseed+node_id)
uniformDistr = RandomDistribution(rng,'uniform',[v_reset,v_thresh])
exc_cells.randomInit(uniformDistr)
inh_cells.randomInit(uniformDistr)

print "%d Connecting populations..." % node_id
connections = {'e2e' : Projection(exc_cells, exc_cells,'fixedProbability', pconn, target='excitatory',rng=rng),
               'e2i' : Projection(exc_cells, inh_cells,'fixedProbability', pconn, target='excitatory',rng=rng),
               'i2e' : Projection(inh_cells, exc_cells,'fixedProbability', pconn, target='inhibitory',rng=rng),
               'i2i' : Projection(inh_cells, inh_cells,'fixedProbability', pconn, target='inhibitory',rng=rng)}

print "%d Setting weights..." % node_id
connections['e2e'].setWeights(w_exc)
connections['e2i'].setWeights(w_exc)
connections['i2e'].setWeights(w_inh)
connections['i2i'].setWeights(w_inh)

#for prj in connections.keys():
#    connections[prj].saveConnections('VAbenchmark_%s_%s_%s.conn' % (benchmark,prj,simulator))

print "%d Number of neurons:     %d excitatory  %d inhibitory" % (node_id, n_exc, n_inh)
print "%d Number of connections: %d e→e  %d e→i  %d i→e  %d i→i" % (node_id, len(connections['e2e']), len(connections['e2i']), len(connections['i2e']), len(connections['i2i']))
print node_id, "Build time:", int(Timer.elapsedTime()), "seconds"
                                                  

# === Setup recording ==========================================================
print "%d Setting up recording..." % node_id
exc_cells.record()
inh_cells.record()
vrecord_list = [exc_cells[0],exc_cells[1]]
exc_cells.record_v(vrecord_list)

# === Run simulation ===========================================================

print "%d Running..." % node_id
Timer.reset()
for i in range(10):
    run(i/10.0*tstop)
print "Run time:", int(Timer.elapsedTime()), "seconds"

print "Mean firing rates (spikes/s): (exc) %4.1f (inh) %4.1f" % \
  (exc_cells.meanSpikeCount()*1000/tstop, inh_cells.meanSpikeCount()*1000/tstop)

# === Print results to file ====================================================

print "%d Writing data to file..." % node_id
Timer.reset()
exc_cells.printSpikes("VAbenchmark_%s_exc_%s.ras" % (benchmark,simulator))
inh_cells.printSpikes("VAbenchmark_%s_inh_%s.ras" % (benchmark,simulator))
exc_cells.print_v("VAbenchmark_%s_exc_%s.v" % (benchmark,simulator),compatible_output=True)
print "Time to print spikes:", int(Timer.elapsedTime()), "seconds"


# === Finished with simulator ==================================================

if "neuron" in simulator: # send e-mail when simulation finished, since it takes ages.
    pyNN.utility.notify("Simulation of Vogels-Abbott %s benchmark with pyNN.%s finished." % (benchmark,simulator))
end()
