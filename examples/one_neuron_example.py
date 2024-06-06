"""
A very simple example of simulating One Neuron
using NEST simulator backend with the help of PyNN.

"""
import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel

sim.setup(timestep=0.1) # (ms)

cell_params = {
    "v_rest": -70.0, # (mV)
    "v_reset": -70.0, # (mV)
    "cm": 0.250, # (nF)
    "tau_m": 10, # (ms)
    "tau_refrac": 2, # (ms)
    "tau_syn_E": 2, # (ms)
    "tau_syn_I": 2, # (ms)
    "v_thresh": -55.0, # (mV)
    "i_offset": 0.376, # (nA)
}


cell_type = sim.IF_curr_alpha(**cell_params)
neuron = sim.Population(1, cell_type, label="Neuron 1")
neuron.record("v")

sim.run(1000.0)

data_v = neuron.get_data().segments[0].filter(name="v")[0]

Figure(
    Panel(
        data_v[:,0],
        xticks=True,
        yticks=True,
        xlabel="Time (in ms)",
        ylabel="Membrane Potential (mV)"
    ),
    title="One Neuron",
    annotations="Translating Single Neuron using PyNN"
).show()


