# coding: utf-8
"""

This is an attempt to implement the FACETS review paper benchmark 2:

"The IF network is based on the CUBA model of Vogels & Abbott (J
Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).  For sufficiently
large size (>4000 neurons), this model displays self-sustained
irregular states."

For now, we are using alpha-synapses 'cos NEST doesn't seem to have any
exponential synapses.

NOTE: still need to scale weights so that charge deposited by the alpha
function is the same as would be deposited by an exponential function with the
same time constant.

Andrew Davison, UNIC, CNRS
August 2006

$Id$
"""

import sys
from copy import copy
from NeuroTools.stgen import StGen

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[1]
else:
    simulator = "oldneuron"    # run using nrngui -python
exec("from pyNN.%s import *" % simulator)

# === Define parameters ========================================================

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 50    # (ms) duration of random stimulation

dt       = 0.1   # (ms) simulation timestep
tstop    = 3000  # (ms) simulaton duration

# Cell parameters
area     = 20000 # (µm²)
tau_m    = 20    # (ms)
cm       = 1     # (µF/cm²)
g_leak   = 5e-5  # (S/cm²)
E_leak   = -49   # (mV)
v_thresh = -50   # (mV)
v_reset  = -60   # (mV)
t_refrac = 5     # (ms) (clamped at v_reset)

# Synapse parameters
Gexc     = 0.27  # (nS)
Ginh     = 4.5   # (nS)
Erev_exc = 0     # (mV)
Erev_inh = -80   # (mV)
tau_exc  = 5     # (ms)
tau_inh  = 10    # (ms)

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells   
n_inh = n - n_exc                     # number of inhibitory cells
w_exc = 1e-3*Gexc*(Erev_exc - E_leak) # (nA) weight of excitatory synapses
w_inh = 1e-3*Ginh*(Erev_inh - E_leak) # (nA)
assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

setup(timestep=dt,min_delay=0.1,max_delay=dt)

exc_cell_params = {
    'tau_m'      : tau_m,    'tau_syn'    : tau_exc,  'tau_refrac' : t_refrac,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm}
inh_cell_params = copy(exc_cell_params)
inh_cell_params['tau_syn'] = tau_inh

Timer.start()

exc_cells = Population((n_exc,),"IF_curr_alpha",exc_cell_params,"Excitatory_Cells")
inh_cells = Population((n_inh,),"IF_curr_alpha",inh_cell_params,"Inhibitory_Cells")
print Timer.elapsedTime()

exc_cells.randomInit('uniform',v_reset,v_thresh)
inh_cells.randomInit('uniform',v_reset,v_thresh)

print Timer.elapsedTime()
connections = {'e2e' : Projection(exc_cells, exc_cells,'fixedProbability', pconn),
               'e2i' : Projection(exc_cells, inh_cells,'fixedProbability', pconn),
               'i2e' : Projection(inh_cells, exc_cells,'fixedProbability', pconn),
               'i2i' : Projection(inh_cells, inh_cells,'fixedProbability', pconn)}
print Timer.elapsedTime()
connections['e2e'].setWeights(w_exc)
connections['e2i'].setWeights(w_exc)
connections['i2e'].setWeights(w_inh)
connections['i2i'].setWeights(w_inh)
print Timer.elapsedTime()

print "Number of neurons:     %d excitatory  %d inhibitory" % (n_exc, n_inh)
print "Number of connections: %d e→e  %d e→i  %d i→e  %d i→i" % (len(connections['e2e']),
                 len(connections['e2i']), len(connections['i2e']), len(connections['i2i']))
print "Build time:", int(Timer.elapsedTime()), "seconds"
                                                  

# === Setup recording ==========================================================
exc_cells.record()
inh_cells.record()
vrecord_list = [exc_cells[0],exc_cells[1]]
exc_cells.record_v(vrecord_list)
print Timer.elapsedTime()

# === Run simulation ===========================================================

Timer.reset()
run(tstop)
print "Run time:", int(Timer.elapsedTime()), "seconds"

print "Mean firing rates (spikes/s): (exc) %4.1f (inh) %4.1f" % \
  (exc_cells.meanSpikeCount()*1000/tstop, inh_cells.meanSpikeCount()*1000/tstop)

# === Print results to file ====================================================

exc_cells.printSpikes("VAbenchmark2_exc_%s.ras" % simulator)
inh_cells.printSpikes("VAbenchmark2_inh_%s.ras" % simulator)
exc_cells.print_v("VAbenchmark2_exc_%s.v" % simulator)
print Timer.elapsedTime()

# === Finished with simulator ==================================================
end()
