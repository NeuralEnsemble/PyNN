"""
Conversion of NEST example script one_neuron_with_noise.py
"""

import pyNN.nest as sim 
from pyNN.utility.plotting import Figure, Panel

sim.setup()

parameters = {
 'v_rest': -70,
 'cm': 0.25,
 'tau_m': 10.0,
 'tau_refrac': 2.0,
 'tau_syn_E': 2.0,
 'tau_syn_I': 2.0,
 'i_offset': 0.376,
 'v_reset': -70.0,
 'v_thresh': -55.0
}


neuron = sim.Population(1, sim.IF_curr_alpha(**parameters), initial_values={"v": -70})
noise = sim.Population(2, sim.SpikeSourcePoisson(rate=[80000, 15000]))

neuron.record("v")


# weight = [0.0012, -0.001]  # nA
weight = 0.0012
delay = 1.0

connections = sim.Projection(
    neuron, noise,
    sim.AllToAllConnector(), 
    sim.StaticSynapse(weight=weight, delay=delay))

sim.run(1000.0)

data_1 = neuron.get_data().segments[0].analogsignals[0]
assert data_1.name == "v"

Figure(
    Panel(
        data_1,
        xticks=True,
        yticks=True
    )
).show()