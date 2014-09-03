"""
Small network created with the Population and Projection classes
and the FromListConnector

Andrew Davison, UNIC, CNRS
April 2013

"""

import numpy
from pyNN.utility import get_script_args, init_logging, normalized_filename

init_logging(None, debug=True)

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

from pyNN.parameters import Sequence

# === Define parameters ========================================================

n = 5    # Number of cells
w = 0.5   # synaptic weight (nA)
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

cells = Population(n, IF_curr_alpha(**cell_params), initial_values={'v': 0.0}, label="cells")

number = int(2*simtime*input_rate/1000.0)
numpy.random.seed(26278342)
def generate_spike_times(i):
    gen = lambda: Sequence(numpy.add.accumulate(numpy.random.exponential(1000.0/input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()
assert generate_spike_times(0).max() > simtime

spike_source = Population(n, SpikeSourceArray(spike_times=generate_spike_times))

spike_source.record('spikes')
cells.record('spikes')
cells[0:1].record('v')

connector = FromListConnector([
    (0, 1, w, syn_delay),
    (0, 2, w, syn_delay),
    (0, 4, w, syn_delay),
    (1, 0, w, syn_delay),
    (1, 1, w, syn_delay),
    (1, 3, w, syn_delay),
    (1, 4, w, syn_delay),
    (2, 3, w, syn_delay),
    (3, 0, w, syn_delay),
    (3, 2, w, syn_delay),
    (4, 2, w, syn_delay),
])
input_conns = Projection(spike_source, cells, connector, StaticSynapse())

# === Run simulation ===========================================================

run(simtime)

#spike_source.write_data("Results/small_network_input_np%d_%s.pkl" % (num_processes(), simulator_name))
filename = normalized_filename("Results", "specific_network", "pkl",
                               simulator_name, num_processes())
cells.write_data(filename, annotations={'script_name': __file__})

print("Mean firing rate: ", cells.mean_spike_count()*1000.0/simtime, "Hz")

# === Clean up and quit ========================================================

end()
