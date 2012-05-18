"""
Small network created with the Population and Projection classes

Andrew Davison, UNIC, CNRS
May 2006

$Id$

"""

import numpy
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

from pyNN.parameters import Sequence

# === Define parameters ========================================================

n = 5    # Number of cells
w = 0.3   # synaptic weight (nA)
cell_params = {
    'tau_m'      : 20.0, # (ms)
    'tau_syn_E'  : 2.0,  # (ms)
    'tau_syn_I'  : 4.0,  # (ms)
    'tau_refrac' : 2.0,  # (ms)
    'v_rest'     : 0.0,  # (mV)
    'v_reset'    : 0.0,  # (mV)
    'v_thresh'   : 20.0, # (mV)
    'cm'         : 0.5}  # (nF)
dt         = 0.1         # (ms)
syn_delay  = 1.0         # (ms)
input_rate = 50.0       # (Hz)
simtime    = 1000.0      # (ms)

# === Build the network ========================================================

setup(timestep=dt, max_delay=syn_delay)

cells = Population(n, IF_curr_alpha, cell_params, initial_values={'v': 0.0}, label="cells")

number = int(2*simtime*input_rate/1000.0)
numpy.random.seed(26278342)
def generate_spike_times(i):
    gen = lambda: Sequence(numpy.add.accumulate(numpy.random.exponential(1000.0/input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()
assert generate_spike_times(0).max() > simtime

spike_source = Population(n, SpikeSourceArray, {'spike_times': generate_spike_times})

spike_source.record('spikes')
cells.record('spikes')
cells[0:1].record('v')

input_conns = Projection(spike_source, cells, AllToAllConnector())
input_conns.setWeights(w)
input_conns.setDelays(syn_delay)

# === Run simulation ===========================================================

run(simtime)

#spike_source.write_data("Results/small_network_input_np%d_%s.pkl" % (num_processes(), simulator_name))
cells.write_data("Results/small_network_np%d_%s.pkl" % (num_processes(), simulator_name))

print "Mean firing rate: ", cells.mean_spike_count()*1000.0/simtime, "Hz"

# === Clean up and quit ========================================================

end()
