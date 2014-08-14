"""
A selection of Izhikevich neurons.

Run as:

$ python Izhikevich.py <simulator>

where <simulator> is 'neuron', 'nest', etc.

"""

from pyNN.utility import get_script_args
from numpy import arange

simulator_name = get_script_args(1)[0]
exec("import pyNN.%s as sim" % simulator_name)

sim.setup(timestep=0.01, min_delay=1.0)

neurons = sim.Population(2, sim.Izhikevich(a=0.02, b=0.2, c=-65, d=6, i_offset=[14.0, 0.0]))

spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=arange(10.0, 51 , 1)))

connection = sim.Projection(spike_source, neurons[1:2], sim.OneToOneConnector(),
                            sim.StaticSynapse(weight=3.0, delay=1.0),
                            receptor_type='excitatory'),

neurons.record('v')

neurons.initialize(v=-70.0, u=-14.0)

sim.run(100.0)

neurons.write_data("Results/Izhikevich_%s.pkl" % simulator_name)

sim.end()
