"""
Conversion of NEST example script twoneurons.py
"""

import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel


sim.setup()


parameters_1 = {
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
parameters_2 = parameters_1.copy()
parameters_2["i_offset"] = 0.0

neuron_1 = sim.Population(1, sim.IF_curr_alpha(**parameters_1), initial_values={"v": -70})
neuron_2 = sim.Population(1, sim.IF_curr_alpha(**parameters_2), initial_values={"v": -70})

neuron_1.record("v")
neuron_2.record("v")

weight = 0.02  # nA
delay = 1.0

connections = sim.Projection(
    neuron_1, neuron_2, 
    sim.AllToAllConnector(), 
    sim.StaticSynapse(weight=weight, delay=delay))

sim.run(1000.0)

data_1 = neuron_1.get_data().segments[0].analogsignals[0]
data_2 = neuron_2.get_data().segments[0].analogsignals[0]

assert data_1.name == "v"

Figure(
    Panel(
        data_1,
        xticks=True,
        yticks=True
    ),
    Panel(
        data_2,
        xticks=True,
        yticks=True
    )
).show()

