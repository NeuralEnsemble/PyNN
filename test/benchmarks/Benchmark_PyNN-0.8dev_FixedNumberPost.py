# coding: utf-8
"""
This script is a modified version of http://neuralensemble.org/trac/PyNN/wiki/Examples/VogelsAbbott

    An implementation of benchmarks 1 and 2 from
        Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398
    simulator_name
    The IF network is based on the CUBA and COBA models of Vogels & Abbott
    (J. Neurosci, 2005).  The model consists of a network of excitatory and
    inhibitory neurons, connected via current-based "exponential"
    synapses (instantaneous rise, exponential decay).

    Andrew Davison, UNIC, CNRS
    August 2006

Author: Bernhard Kaplan, bkaplan@kth.se
"""
import time
t0 = time.time()

# to store timing information
from pyNN.utility import Timer
timer = Timer()
timer.start()
times = {}
times['t_startup'] = time.time() - t0

# check imports
import numpy as np
import os
import socket
from math import *
import json
from pyNN.utility import get_script_args
simulator_name = 'nest'
from pyNN.nest import *
#exec("from pyNN.%s import *" % simulator_name)
try:
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    node_id, n_proc = comm.rank, comm.size
    print("USE_MPI:", USE_MPI, 'pc_id, n_proc:', node_id, n_proc)
except:
    USE_MPI = False
    node_id, n_proc, comm = 0, 1, None
    print("MPI not used")

from pyNN.random import NumpyRNG, RandomDistribution
times['t_import'] = timer.diff()

# === DEFINE PARAMETERS
benchmark = "COBA"
rngseed = 98765
parallel_safe = True
np = num_processes()
folder_name = 'Results_PyNN_FixedNumberPost_np%d/' % (np)
gather = False  # gather spikes and membrane potentials on one process
times_fn = 'pynn_times_FixedNumberPost_gather%d_np%d.dat' % (gather, np)

n_cells = 200 * np
r_ei = 4.0   # number of excitatory cells:number of inhibitory cells
n_exc = int(round((n_cells * r_ei / (1 + r_ei))))  # number of excitatory cells
n_inh = n_cells - n_exc                     # number of inhibitory cells
n_cells_to_record = np
n_conn_out = 1000  # number of outgoing connections per neuron
weight = 1e-8  # connection weights

f_noise_exc = 3000.
f_noise_inh = 2000.
w_noise_exc = 1e-3
w_noise_inh = 1e-3
dt = 0.1   # (ms) simulation timestep
t_sim = 1000  # (ms) simulaton duration
delay = 1 * dt

# === SETUP
node_id = setup(timestep=dt, min_delay=delay, max_delay=delay)
times['t_setup'] = timer.diff()

host_name = socket.gethostname()
print("Host #%d is on %s" % (node_id + 1, host_name))


# === CREATE
print("%s Creating cell populations..." % node_id)
#celltype = IF_cond_exp
exc_cells = Population(n_exc, IF_cond_exp(), label="Excitatory_Cells")
inh_cells = Population(n_inh, IF_cond_exp(), label="Inhibitory_Cells")
times['t_create'] = timer.diff()

print("Creating noise sources ...")
exc_noise_in_exc = Population(n_exc, SpikeSourcePoisson, {'rate': f_noise_exc})
inh_noise_in_exc = Population(n_exc, SpikeSourcePoisson, {'rate': f_noise_inh})
exc_noise_in_inh = Population(n_inh, SpikeSourcePoisson, {'rate': f_noise_exc})
inh_noise_in_inh = Population(n_inh, SpikeSourcePoisson, {'rate': f_noise_inh})
times['t_create_noise'] = timer.diff()

print("%s Initialising membrane potential to random values..." % node_id)
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', [-50, -70], rng=rng)
exc_cells.initialize(v=uniformDistr)
inh_cells.initialize(v=uniformDistr)
times['t_vinit'] = timer.diff()

print("%s Connecting populations..." % node_id)
ee_conn = FixedNumberPostConnector(n_conn_out)  # , weights=weight, delays=delay)
ei_conn = FixedNumberPostConnector(n_conn_out)  # , weights=weight, delays=delay)
ie_conn = FixedNumberPostConnector(n_conn_out)  # , weights=weight, delays=delay)
ii_conn = FixedNumberPostConnector(n_conn_out)  # , weights=weight, delays=delay)
times['t_connector'] = timer.diff()

connections = {}
connections['e2e'] = Projection(exc_cells, exc_cells, ee_conn, receptor_type='excitatory')  # , rng=rng)
connections['e2i'] = Projection(exc_cells, inh_cells, ei_conn, receptor_type='excitatory')  # , rng=rng)
connections['i2e'] = Projection(inh_cells, exc_cells, ie_conn, receptor_type='inhibitory')  # , rng=rng)
connections['i2i'] = Projection(inh_cells, inh_cells, ii_conn, receptor_type='inhibitory')  # , rng=rng)
connections['e2e'].set(weight=weight)
connections['e2i'].set(weight=weight)
connections['i2e'].set(weight=weight)
connections['i2i'].set(weight=weight)
times['t_projection'] = timer.diff()

exc_noise_connector = OneToOneConnector()
inh_noise_connector = OneToOneConnector()

noise_ee_prj = Projection(exc_noise_in_exc, exc_cells, exc_noise_connector, receptor_type="excitatory")
noise_ei_prj = Projection(exc_noise_in_inh, inh_cells, exc_noise_connector, receptor_type="excitatory")
noise_ie_prj = Projection(inh_noise_in_exc, exc_cells, inh_noise_connector, receptor_type="inhibitory")
noise_ii_prj = Projection(inh_noise_in_inh, inh_cells, inh_noise_connector, receptor_type="inhibitory")
noise_ee_prj.set(weight=w_noise_exc)
noise_ei_prj.set(weight=w_noise_exc)
noise_ie_prj.set(weight=w_noise_inh)
noise_ii_prj.set(weight=w_noise_inh)
times['t_connect_noise'] = timer.diff()


# === Setup recording ==========================================================
print("%s Setting up recording..." % node_id)
exc_cells.record('spikes')
inh_cells.record('spikes')
cells_to_record = range(n_cells_to_record)
exc_cells[list(cells_to_record)].record_v()
times['t_record'] = timer.diff()


print("%d Print(exc spikes to file..." % node_id)
if not(os.path.isdir(folder_name)) and (rank() == 0):
    os.mkdir(folder_name)

# === Save connections to file =================================================
#print("%s Saving projections ..." % node_id
#for prj in connections.keys():
#    connections[prj].saveConnections('%s/VAbenchmark_%s_%s_%s_np%d.conn' % (folder_name, benchmark, prj, simulator_name, np))
#times['t_save_connections'] = timer.diff()


# === Run simulation ===========================================================
print("%d Running simulation..." % node_id)
run(t_sim)
times['t_run'] = timer.diff()


# === Print(results to file ====================================================
exc_spike_fn = "%s/VAbenchmark_%s_exc_%s_np%d_%d.pkl" % (folder_name, benchmark, simulator_name, np, node_id)
exc_cells.printSpikes(exc_spike_fn, gather=gather)
print("%d Print(inh spikes to file..." % node_id)
inh_spike_fn = "%s/VAbenchmark_%s_inh_%s_np%d_%d.pkl" % (folder_name, benchmark, simulator_name, np, node_id)
inh_cells.printSpikes(inh_spike_fn, gather=gather)
print("%d Print(voltage to file..." % node_id)
exc_cells[list(cells_to_record)].print_v("%s/VAbenchmark_%s_exc_%s_np%d_%d.pkl" % (folder_name, benchmark, simulator_name, np, node_id), gather=gather)
times['t_printSpikes'] = timer.diff()

# === Load spike file and calculate conductances ====================
#exc_spikes = np.loadtxt(exc_spike_fn)
#E_count = exc_cells.meanSpikeCount()
#I_count = inh_cells.meanSpikeCount()
#f_exc = E_count*1000.0/t_sim
#f_inh = I_count*1000.0/t_sim
#g_ee = f_exc * w_ee * tau_exc * connections['e2e'].size() / n_exc
#g_ei = f_exc * w_ei * tau_exc * connections['e2i'].size() / n_exc
#g_ie = f_inh * w_ie * tau_inh * connections['i2e'].size() / n_inh
#g_ii = f_inh * w_ii * tau_inh * connections['i2i'].size() / n_inh

connections_string = "%d e→e  %d e→i  %d i→e  %d i→i" % (connections['e2e'].size(),
                                                  connections['e2i'].size(),
                                                  connections['i2e'].size(),
                                                  connections['i2i'].size())
n_total = connections['e2e'].size() + connections['e2i'].size() + connections['i2e'].size() + connections['i2i'].size()

n_connections = connections['e2e'].size() + connections['e2i'].size() + connections['i2e'].size() + connections['i2i'].size()
n_conn_ee, n_conn_ei, n_conn_ie, n_conn_ii = connections['e2e'].size(), connections['e2i'].size(), connections['i2e'].size(), connections['i2i'].size()
times['t_analysis'] = timer.diff()

if node_id == 0:
    print("\n--- Vogels-Abbott Network Simulation ---")
    print("Nodes                  : %d" % np)
    print("Simulation type        : %s" % benchmark)
    print("Simulator name         : %s" % simulator_name)
    print("Number of Neurons      : %d  n_exc %d  n_inh %d" % (n_cells, n_exc, n_inh))
    print("Number of Synapses     : %s" % connections_string)
    print("Total Num Synapses     : %s" % n_total)
#    print("Excitatory rate        : %g Hz" % f_exc)
#    print("Inhibitory rate        : %g Hz" % f_inh)
#    print(timing_info)

print("%d PyNN end" % node_id)
end()
print("%d PyNN end finish" % node_id)
times['t_end'] = timer.diff()

if node_id == 0:
    t_all = 0.
    for k in times.keys():
        t_all += times[k]
    times['t_sum'] = t_all
    times['n_exc'] = n_exc
    times['n_inh'] = n_inh
    times['n_cells'] = n_cells
    times['n_proc'] = np

    times['n_ee'] = n_conn_ee
    times['n_ei'] = n_conn_ei
    times['n_ie'] = n_conn_ie
    times['n_ii'] = n_conn_ii
    times['n_connections'] = n_connections

    f = file(times_fn, 'w')
    json.dump(times, f)

