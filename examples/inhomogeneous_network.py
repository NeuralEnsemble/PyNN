"""
Small, inhomogeneous network

Andrew Davison, UNIC, CNRS
December 2012

"""

from pyNN.utility import init_logging, normalized_filename
from pyNN.parameters import Sequence
from pyNN.space import Grid2D
from importlib import import_module
import numpy
from lazyarray import sqrt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('simulator_name')
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

init_logging(None, debug=args.debug)

sim = import_module("pyNN.%s" % args.simulator_name)

simtime = 100.0
input_rate = 20.0
n_cells = 9

sim.setup()

cell_type = sim.IF_cond_exp(tau_m=10.0,
#                            v_rest=lambda x, y, z: -60.0 - sqrt((x**2 + y**2)/100),
#                            v_thresh=lambda x, y, z: -55.0 + x/10.0)
                            v_rest=lambda i: -60.0 + i,
                            v_thresh=lambda i: -55.0 + i)

cells = sim.Population(n_cells, cell_type,
                       structure=Grid2D(dx=100.0, dy=100.0),
                       initial_values={'v': lambda i: -60.0 - i},
                       label="cells")

print("positions:")
print(cells.positions)

for name in ('tau_m', 'v_rest', 'v_thresh'):
    print(name, "=", cells.get(name))

number = int(2*simtime*input_rate/1000.0)
numpy.random.seed(26278342)
def generate_spike_times(i):
    gen = lambda: Sequence(numpy.add.accumulate(numpy.random.exponential(1000.0/input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()
assert generate_spike_times(0).max() > simtime

spike_source = sim.Population(n_cells, sim.SpikeSourceArray(spike_times=generate_spike_times))


connections = sim.Projection(spike_source, cells,
                             sim.FixedProbabilityConnector(0.5),
                             sim.StaticSynapse(weight='1/(1+d)',
                                               delay=0.5)
                            )

print("weights:")
print(str(connections.get('weight', format='array')).replace('nan', ' . '))
print("delays:")
print(str(connections.get('delay', format='array')).replace('nan', ' . '))

cells.record(['spikes', 'v'])


sim.run(100.0)

filename = normalized_filename("Results", "inhomogeneous_network", "pkl",
                               args.simulator_name)
cells.write_data(filename, annotations={'script_name': __file__})

print("Mean firing rate: ", cells.mean_spike_count()*1000.0/sim.get_current_time(), "Hz")

sim.end()
