"""
Conversion of NEST example script oneneuron.py
"""

import pyNN.nest as sim
from pyNN.utility.plotting import Panel, Figure

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
neuron.record("v")

sim.run(200.0)


data_1 = neuron.get_data().segments[0].analogsignals[0]

assert data_1.name == "v"

Figure(
    Panel(
        data_1,
        xticks=True,
        yticks=True
    )
).show()


neuron.set(i_offset=0.450)


sim.run(1000.0)

data_2 = neuron.get_data().segments[0].analogsignals[0]

assert data_2.name == "v"

Figure(
    
    Panel(
        data_2,
        xticks=True,
        yticks=True
    )
).show()